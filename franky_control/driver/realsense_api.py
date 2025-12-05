import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict
import os
import threading
import time
from PIL import Image

class RealsenseAPI:
    """Wrapper that implements boilerplate code for RealSense cameras
    
    Features:
    - Async frame capture in background thread for non-blocking reads
    - Double-buffered frames for lock-free fast access in control loops
    """

    def __init__(self, height=480, width=640, fps=30, warm_start=60, use_async=True):
        self.height = height
        self.width = width
        self.fps = fps
        self.use_async = use_async

        # Identify devices
        self.device_ls = []
        for c in rs.context().query_devices():
            self.device_ls.append(c.get_info(rs.camera_info.serial_number))

        # Start stream
        print(f"Connecting to RealSense cameras ({len(self.device_ls)} found) ...")
        self.pipes = []
        self.profiles = OrderedDict()
        self.sensors = OrderedDict()  # Store sensors for parameter control

        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            profile = pipe.start(config)
            self.profiles[device_id] = profile
            
            device = profile.get_device()
            
            depth_sensor = device.first_depth_sensor()
            color_sensor = device.first_color_sensor()

            if depth_sensor and depth_sensor.supports(rs.option.frames_queue_size):
                depth_sensor.set_option(rs.option.frames_queue_size, 1)
            
            if color_sensor and color_sensor.supports(rs.option.frames_queue_size):
                color_sensor.set_option(rs.option.frames_queue_size, 1)

            self.sensors[device_id] = {
                'color': color_sensor,
                'depth': depth_sensor,
                'device': device
            }

            print(f"Connected to camera {i+1} ({device_id}).")

        self.align = rs.align(rs.stream.color)
        
        # Double-buffered async frame cache for lock-free reads
        # We use two buffers: one for writing (back), one for reading (front)
        # This eliminates lock contention between capture thread and main thread
        num_cams = len(self.device_ls)
        self._rgb_buffer_0 = np.zeros([num_cams, self.height, self.width, 3], dtype=np.uint8)
        self._rgb_buffer_1 = np.zeros([num_cams, self.height, self.width, 3], dtype=np.uint8)
        self._depth_buffer_0 = np.zeros([num_cams, self.height, self.width], dtype=np.uint16)
        self._depth_buffer_1 = np.zeros([num_cams, self.height, self.width], dtype=np.uint16)
        self._active_buffer = 0  # 0 or 1, indicates which buffer is ready for reading
        self._timestamp_0 = 0.0
        self._timestamp_1 = 0.0
        self._buffer_lock = threading.Lock()  # Only used for buffer swap, very fast
        
        self._async_thread = None
        self._async_running = False
        
        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        # Also populate initial cache with valid frames
        for i in range(warm_start):
            framesets = self._get_frames()
            # On last warm start iteration, populate the cache with valid frames
            if i == warm_start - 1:
                for cam_idx, frameset in enumerate(framesets):
                    color_frame = frameset.get_color_frame()
                    depth_frame = frameset.get_depth_frame()
                    self._rgb_buffer_0[cam_idx, :, :, :] = np.asanyarray(color_frame.get_data())
                    self._depth_buffer_0[cam_idx, :, :] = np.asanyarray(depth_frame.get_data())
                    self._rgb_buffer_1[cam_idx, :, :, :] = self._rgb_buffer_0[cam_idx, :, :, :]
                    self._depth_buffer_1[cam_idx, :, :] = self._depth_buffer_0[cam_idx, :, :]
                self._timestamp_0 = time.time()
                self._timestamp_1 = self._timestamp_0
        
        # Start async capture if enabled
        if self.use_async:
            self.start_async_capture()
    
    def start_async_capture(self):
        """Start background thread for async frame capture."""
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        
        self._async_running = True
        self._async_thread = threading.Thread(target=self._async_capture_loop, daemon=True)
        self._async_thread.start()
        print(f"[Camera] Async capture started ({self.fps} FPS target, double-buffered)")
    
    def stop_async_capture(self):
        """Stop async capture thread."""
        self._async_running = False
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)
            self._async_thread = None
    
    def get_frame_age(self) -> float:
        """Get age of cached frame in seconds.
        
        Returns:
            Time since the cached frame was captured (seconds).
            Returns 0 if async capture is not running.
        """
        if not self._async_running:
            return 0.0
        # Read from active buffer (lock-free read of atomic int)
        active = self._active_buffer
        timestamp = self._timestamp_0 if active == 0 else self._timestamp_1
        if timestamp == 0:
            return 0.0
        return time.time() - timestamp
    
    def get_frame_timestamp(self) -> float:
        """Get timestamp of cached frame.
        
        Returns:
            Unix timestamp when the cached frame was captured.
        """
        active = self._active_buffer
        return self._timestamp_0 if active == 0 else self._timestamp_1
    
    def _async_capture_loop(self):
        """Background loop to continuously capture frames using double-buffering.
        
        Uses double-buffering to eliminate lock contention:
        - Write to back buffer while front buffer is being read
        - Swap buffers atomically when write is complete
        
        Critical: We always write to the OPPOSITE buffer of what's being read.
        The buffer swap happens AFTER all writes are complete.
        """
        while self._async_running:
            try:
                framesets = [pipe.wait_for_frames() for pipe in self.pipes]
                aligned_frames = [self.align.process(frameset) for frameset in framesets]
                capture_time = time.time()
                
                # Read active buffer ONCE at the start - this tells us which buffer is being read
                # We write to the OTHER buffer
                reading_buffer = self._active_buffer
                
                if reading_buffer == 0:
                    # Main thread reads buffer 0, we write to buffer 1
                    for i, frameset in enumerate(aligned_frames):
                        color_frame = frameset.get_color_frame()
                        depth_frame = frameset.get_depth_frame()
                        self._rgb_buffer_1[i, :, :, :] = np.asanyarray(color_frame.get_data())
                        self._depth_buffer_1[i, :, :] = np.asanyarray(depth_frame.get_data())
                    self._timestamp_1 = capture_time
                    # Now that write is complete, swap to make buffer 1 active for reading
                    self._active_buffer = 1
                else:
                    # Main thread reads buffer 1, we write to buffer 0
                    for i, frameset in enumerate(aligned_frames):
                        color_frame = frameset.get_color_frame()
                        depth_frame = frameset.get_depth_frame()
                        self._rgb_buffer_0[i, :, :, :] = np.asanyarray(color_frame.get_data())
                        self._depth_buffer_0[i, :, :] = np.asanyarray(depth_frame.get_data())
                    self._timestamp_0 = capture_time
                    # Now that write is complete, swap to make buffer 0 active for reading
                    self._active_buffer = 0
                    
            except Exception as e:
                pass  # Silently ignore errors
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles.values():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()

            intrinsics_ls.append(intrinsics)

        return intrinsics_ls

    def get_intrinsics_dict(self):
        intrinsics_ls = OrderedDict()
        for device_id, profile in self.profiles.items():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            param_dict = dict([(p, getattr(intrinsics, p)) for p in dir(intrinsics) if not p.startswith('__')])
            param_dict['model'] = param_dict['model'].name

            intrinsics_ls[device_id] = param_dict

        return intrinsics_ls
    
    def get_num_cameras(self):
        return len(self.device_ls)

    def get_rgbd(self):
        """Returns a numpy array of [n_cams, height, width, RGBD]
        
        Uses double-buffered frames if async capture is running (lock-free).
        """
        if self._async_running:
            # Read from active buffer (no lock needed - double buffering)
            active = self._active_buffer
            if active == 0:
                rgb = self._rgb_buffer_0
                depth = self._depth_buffer_0
            else:
                rgb = self._rgb_buffer_1
                depth = self._depth_buffer_1
            
            rgbd = np.empty([self.get_num_cameras(), self.height, self.width, 4], dtype=np.uint16)
            rgbd[:, :, :, :3] = rgb
            rgbd[:, :, :, 3] = depth
            return rgbd
        
        # Fallback to sync read
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        rgbd = np.empty([num_cams, self.height, self.width, 4], dtype=np.uint16)

        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgbd[i, :, :, :3] = np.asanyarray(color_frame.get_data())

            depth_frame = frameset.get_depth_frame()
            rgbd[i, :, :, 3] = np.asanyarray(depth_frame.get_data())

        return rgbd

    def get_rgb(self):
        """Returns a numpy array of [n_cams, height, width, RGB]
        
        Uses double-buffered frames if async capture is running (lock-free, ~0.5ms).
        """
        if self._async_running:
            # Read from active buffer (no lock needed - double buffering)
            active = self._active_buffer
            if active == 0:
                return self._rgb_buffer_0.copy()
            else:
                return self._rgb_buffer_1.copy()
        
        # Fallback to sync read
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        rgb = np.empty([num_cams, self.height, self.width, 3], dtype=np.uint8)

        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgb[i, :, :, :] = np.asanyarray(color_frame.get_data())

        return rgb

    def get_depth(self):
        """Returns a numpy array of [n_cams, height, width, depth]
        
        Uses double-buffered frames if async capture is running (lock-free).
        """
        if self._async_running:
            # Read from active buffer (no lock needed - double buffering)
            active = self._active_buffer
            if active == 0:
                return self._depth_buffer_0.copy()
            else:
                return self._depth_buffer_1.copy()
        
        # Fallback to sync read
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        depth = np.empty([num_cams, self.height, self.width], dtype=np.uint16)

        for i, frameset in enumerate(framesets):
            depth_frame = frameset.get_depth_frame()
            depth[i, :, :] = np.asanyarray(depth_frame.get_data())

        return depth

    def get_key_camera_params(self, device_index=0):
        """Get key camera parameters for filename generation."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            depth_sensor = self.sensors[device_id]['depth']
            
            params = {}
            
            # Color sensor parameters
            try:
                if color_sensor.supports(rs.option.exposure):
                    params['exp'] = int(color_sensor.get_option(rs.option.exposure))
                if color_sensor.supports(rs.option.gain):
                    params['gain'] = int(color_sensor.get_option(rs.option.gain))
                if color_sensor.supports(rs.option.white_balance):
                    params['wb'] = int(color_sensor.get_option(rs.option.white_balance))
                
                # Check auto exposure status
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    params['auto_exp'] = bool(color_sensor.get_option(rs.option.enable_auto_exposure))
                
                # Check auto white balance status
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    params['auto_wb'] = bool(color_sensor.get_option(rs.option.enable_auto_white_balance))
                    
            except Exception:
                pass
            
            # Depth sensor parameters
            try:
                if depth_sensor.supports(rs.option.laser_power):
                    params['laser'] = int(depth_sensor.get_option(rs.option.laser_power))
            except Exception:
                pass
            
            return params
            
        except Exception as e:
            print(f"Failed to get camera parameters for camera {device_index+1}: {e}")
            return {}

    def get_all_cameras_params(self):
        """Get key parameters for all cameras."""
        all_params = {}
        for i in range(self.get_num_cameras()):
            all_params[f'cam{i+1}'] = self.get_key_camera_params(i)
        return all_params

    # New camera parameter control methods
    def set_exposure(self, device_index=0, exposure_value=None):
        """Set exposure for color sensor. If None, enables auto exposure."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            
            if exposure_value is None:
                # Enable auto exposure
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print(f"Camera {device_index+1}: Auto exposure enabled")
            else:
                # Disable auto exposure and set manual value
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, exposure_value)
                print(f"Camera {device_index+1}: Manual exposure set to {exposure_value}")
        except Exception as e:
            print(f"Failed to set exposure for camera {device_index+1}: {e}")

    def set_gain(self, device_index=0, gain_value=None):
        """Set gain for color sensor. If None, enables auto gain."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            
            if gain_value is None:
                # Enable auto gain
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print(f"Camera {device_index+1}: Auto gain enabled")
            else:
                # Set manual gain
                color_sensor.set_option(rs.option.gain, gain_value)
                print(f"Camera {device_index+1}: Gain set to {gain_value}")
        except Exception as e:
            print(f"Failed to set gain for camera {device_index+1}: {e}")

    def set_laser_power(self, device_index=0, power_value=150):
        """Set laser power for depth sensor (0-360)."""
        try:
            device_id = self.device_ls[device_index]
            depth_sensor = self.sensors[device_id]['depth']
            
            depth_sensor.set_option(rs.option.laser_power, power_value)
            print(f"Camera {device_index+1}: Laser power set to {power_value}")
        except Exception as e:
            print(f"Failed to set laser power for camera {device_index+1}: {e}")

    def set_white_balance(self, device_index=0, wb_value=None):
        """Set white balance for color sensor. If None, enables auto white balance."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            
            if wb_value is None:
                # Enable auto white balance
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                print(f"Camera {device_index+1}: Auto white balance enabled")
            else:
                # Disable auto and set manual value
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                color_sensor.set_option(rs.option.white_balance, wb_value)
                print(f"Camera {device_index+1}: White balance set to {wb_value}")
        except Exception as e:
            print(f"Failed to set white balance for camera {device_index+1}: {e}")

    def get_camera_options(self, device_index=0):
        """Get available options and their current values for a camera."""
        try:
            device_id = self.device_ls[device_index]
            sensors = self.sensors[device_id]
            
            options_info = {}
            for sensor_type, sensor in sensors.items():
                if sensor_type == 'device':
                    continue
                options_info[sensor_type] = {}
                for option in sensor.get_supported_options():
                    try:
                        if not sensor.is_option_read_only(option):
                            current_value = sensor.get_option(option)
                            option_range = sensor.get_option_range(option)
                            options_info[sensor_type][option.name] = {
                                'current': current_value,
                                'min': option_range.min,
                                'max': option_range.max,
                                'step': option_range.step,
                                'default': option_range.default
                            }
                    except Exception:
                        continue
            
            return options_info
        except Exception as e:
            print(f"Failed to get options for camera {device_index+1}: {e}")
            return {}

    def close(self):
        """Properly close all camera streams."""
        # Stop async capture first
        self.stop_async_capture()
        
        for pipe in self.pipes:
            pipe.stop()
        print("All camera streams closed.")

if __name__ == "__main__":
    cams = RealsenseAPI()

    print(f"Num cameras: {cams.get_num_cameras()}")
    
    # # Test camera options
    # if cams.get_num_cameras() > 0:
    #     options = cams.get_camera_options(0)
    #     print("Camera 0 options:", options)
        
    #     # Test key parameters
    #     key_params = cams.get_key_camera_params(0)
    #     print("Camera 0 key params:", key_params)
    
    intrinsics_list = cams.get_intrinsics()
    print("\n[1] Intrinsics list returned by get_intrinsics() (rs.intrinsics objects):")
    print(intrinsics_list)
    print("\n[2] Detailed parameters for each camera:")
    for cam_idx, intrinsics in enumerate(intrinsics_list):
        print(f"\nIntrinsics for camera {cam_idx + 1}")
        print(f"Resolution (width x height): {intrinsics.width}x{intrinsics.height}")
        print(f"Focal length fx: {intrinsics.fx:.6f}")
        print(f"Focal length fy: {intrinsics.fy:.6f}")
        print(f"Principal point ppx: {intrinsics.ppx:.6f}")
        print(f"Principal point ppy: {intrinsics.ppy:.6f}")
        print(f"Distortion model: {intrinsics.model.name}")
        print(f"Distortion coefficients: {[round(c,6) for c in intrinsics.coeffs]}")

    rgbd = cams.get_rgbd()
    rgb = cams.get_rgb()
    out_dir = "cameras_output"
    os.makedirs(out_dir, exist_ok=True)

    for cam_idx in range(cams.get_num_cameras()):
        image = Image.fromarray(rgb[cam_idx].astype(np.uint8))
        image.save(f"{out_dir}/camera_{cam_idx + 1}.jpg")
    
    cams.close()
