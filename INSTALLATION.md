# INSTALLATION

## Env Creation
```bash
conda create -n franky-control python==3.10 -y
conda activate franky-control
```

## ⚙️ Setup

To install franky, you have to follow three steps:

1. Ensure that you are using a real-time kernel
2. Ensure that the executing user has permission to run real-time applications
3. Install franky via pip or build it from source

### Installing a real-time kernel

In order for Franky to function properly, it requires the underlying OS to use a real-time kernel.
Otherwise, you might see `communication_constrains_violation` errors.

To check whether your system is currently using a real-time kernel, type `uname -a`.
You should see something like this:

```
$ uname -a
Linux [PCNAME] 5.15.0-1056-realtime #63-Ubuntu SMP PREEMPT_RT ...
```

If it does not say PREEMPT_RT, you are not currently running a real-time kernel.

`sudo apt install zstd`

There are multiple ways of installing a real-time kernel.
You can [build it from source](https://frankarobotics.github.io/docs/libfranka/docs/installation_linux.html#setting-up-the-real-time-kernel) or, if you are using Ubuntu, it can be [enabled through Ubuntu Pro](https://ubuntu.com/real-time).

### Allowing the executing user to run real-time applications

First, create a group `realtime` and add your user (or whoever is running franky) to this group:

```bash
sudo addgroup realtime
sudo usermod -a -G realtime $(whoami)
```

Afterward, add the following limits to the real-time group in /etc/security/limits.conf:

```
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
```

Log out and log in again to let the changes take effect.

To verify that the changes were applied, check if your user is in the `realtime` group:

```bash
$ groups
... realtime
```

If real-time is not listed in your groups, try rebooting.

### Installing franky

#### Step 1: Install franky-control (the robot control library)

We provide wheels for libfranka versions *0.7.1*, *0.8.0*, *0.9.2*, *0.12.1*, *0.13.3*,
*0.14.2*, *0.17.0*, and *0.18.0*.

```bash
# Choose the version that matches your Franka system
VERSION=0-9-2   # for Franka Emika Panda system version 4.2.1
VERSION=0-17-0  # for Franka Research 3 system version 5.7.2
VERSION=0-18-0  # for Franka Research 3 system version 5.9.0

# Download and install franky-control wheels
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip uninstall franky-control -y 2>/dev/null  # Remove if previously installed
pip install --no-index --find-links=./dist franky-control
```

#### Step 2: Install this package (franky_control)

```bash
# Basic installation (core dependencies only)
pip install -e .
pip install -e ".[all]"         # Everything

# Or install with specific optional features:
pip install -e ".[spacemouse]"  # SpaceMouse controller support
pip install -e ".[realsense]"   # RealSense camera support  
pip install -e ".[ik]"          # IK solver (PyTorch + SAPIEN)
pip install -e ".[hardware]"    # All hardware components
pip install -e ".[dev]"         # Development tools
```

#### Step 3 (Optional): Install PyTorch with specific index URL

If you need CPU-only PyTorch or a specific version:

```bash
# CPU-only PyTorch (recommended for real robot control)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Or use Tsinghua mirror (for users in China)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cpu
```


## Examples

See the files in [examples](./franky_control/examples/__init__.py)

```python

python -m franky_control.examples.realtime_pose_control

python -m franky_control.examples.reset_arm

```

## Data collection

See the files in [data collection](./franky_control/data_collection/__init__.py)

Full example with all parameters:

```bash
python -m franky_control.data_collection.data_collection_with_ik \
    --task_name "assembly_task" \
    --instruction "assemble the parts together" \
    --robot_ip "172.16.0.2" \
    --min_action_steps 100 \
    --max_action_steps 500 \
    --episode_idx 0 \
    --pos_scale 0.01 \
    --rot_scale 0.01 \
    --use_space_mouse \
    --control_frequency 15.0 \
    --verify_ik \
    --use_cameras
    # --use_joint_filter
```

Test data collection with hardware:

```bash
python -m franky_control.data_collection.data_collection_with_ik \
    --task_name "demo" \
    --instruction "demonstration task" \
    --control_frequency 10.0 \
    --robot_ip "172.16.0.2"
```

```bash
python -m franky_control.data_collection.data_collection_cartesian \
    --task_name "test" \
    --instruction "test task" \
    --robot_ip "172.16.0.2" \
    --control_frequency 10.0 \
    --use_space_mouse


python -m franky_control.data_collection.data_collection_cartesian_relative \
    --task_name "test" \
    --instruction "test task" \
    --robot_ip "172.16.0.2" \
    --control_frequency 10.0 \
    --use_space_mouse
```

Test data collection without hardware:

```bash
python -m franky_control.data_collection.data_collection_with_ik \
    --task_name "test" \
    --instruction "test" \
    --robot_ip "172.16.0.2" \
    --no_use_cameras \
    --use_space_mouse \
    --control_frequency 10.0 \
    --min_action_steps 10
```


## Policy Deploy

See the files in [policy deploy](./franky_control/deploy/__init__.py)


```bash



```

