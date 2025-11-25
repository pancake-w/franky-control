# Franky Python API 完整参考文档

本文档是 `franky` Python API 的完整合并版本，包含了机器人控制、运动、状态、辅助工具、夹爪以及默认参数等所有内容。

---

## 目录

1.  **[第一部分: Robot 类](#part1)**
    *   1.1 初始化
    *   1.2 属性 (Properties)
    *   1.3 运动控制方法
    *   1.4 阻抗控制方法
    *   1.5 碰撞和接触配置
    *   1.6 负载配置
    *   1.7 EE（末端执行器）配置
    *   1.8 错误恢复
    *   1.9 其他方法

2.  **[第二部分: Motion 类](#part2)**
    *   2.1 关节位置控制
    *   2.2 关节速度控制
    *   2.3 笛卡尔位置控制
    *   2.4 笛卡尔速度控制
    *   2.5 运动回调

3.  **[第三部分: 状态类与反应系统](#part3)**
    *   3.1 RobotState - 完整机器人状态
    *   3.2 CartesianState - 笛卡尔状态
    *   3.3 JointState - 关节状态
    *   3.4 辅助类 (RobotPose, RobotVelocity, etc.)
    *   4.1 反应系统: Measure
    *   4.2 反应系统: Condition
    *   4.3 反应系统: Reaction

4.  **[第四部分: 辅助类与Gripper](#part4)**
    *   5.1 Affine - 仿射变换
    *   5.2 Duration - 时间长度
    *   5.3 RelativeDynamicsFactor - 相对动力学因子
    *   5.4 枚举类型 (ReferenceType, RobotMode, etc.)
    *   5.6 BoolFuture - 异步布尔结果
    *   6.1 Gripper - 夹爪控制

5.  **[附录: 默认参数速查表](#appendix)**
    *   1. 动力学限制默认值
    *   2. 阻抗控制默认值
    *   3. 碰撞检测默认值
    *   4. 负载配置默认值
    *   5. Robot初始化参数
    *   6. 常用配置组合

---

<a name="part1"></a>
## 第一部分: Robot 类

`Robot` 类是franky的核心类，用于控制Franka机器人。

### 1.1 初始化

```python
from franky import Robot

# 连接到机器人（需要机器人的IP地址）
robot = Robot("10.90.90.1")

# 或者指定实时配置
robot = Robot("10.90.90.1", realtime_config=RealtimeConfig.EnforceRealtime)
```

**参数:**
- `fci_ip: str` - 机器人的FCI IP地址
- `realtime_config: RealtimeConfig` (可选) - 实时配置模式
  - `RealtimeConfig.EnforceRealtime` - 强制实时模式
  - `RealtimeConfig.IgnoreRealtime` - 忽略实时模式

---

### 1.2 属性 (Properties)

#### 1.2.1 状态相关属性

```python
# 获取完整的机器人状态
state: RobotState = robot.state

# 获取当前笛卡尔状态（位姿和速度）
cartesian_state: CartesianState = robot.current_cartesian_state

# 获取当前关节状态（位置和速度）
joint_state: JointState = robot.current_joint_state
```

**类型说明:**
- `state: RobotState` - 完整的机器人状态，包含50+字段（见第3部分详细说明）
- `current_cartesian_state: CartesianState` - 笛卡尔空间状态
- `current_joint_state: JointState` - 关节空间状态

#### 1.2.2 动力学限制属性

```python
# 全局相对动力学因子（影响速度、加速度、加加速度）
robot.relative_dynamics_factor = 0.05  # 使用最大值的5%

# 或者分别设置速度、加速度、加加速度的缩放因子
robot.relative_dynamics_factor = RelativeDynamicsFactor(
    velocity=0.1,      # 速度使用最大值的10%
    acceleration=0.05, # 加速度使用最大值的5%
    jerk=0.1          # 加加速度使用最大值的10%
)

# ===== 速度限制 (Velocity Limits) =====

# 平移速度限制 [m/s]
# 默认值: 1.7 m/s (Panda和FR3都推荐此默认值)
# 最大值: 3.0 m/s (Panda) / 1.7 m/s (FR3)
robot.translation_velocity_limit.set(1.7)
max_vel = robot.translation_velocity_limit.max  # 获取最大值

# 旋转速度限制 [rad/s]
# 默认值: 2.5 rad/s
# 最大值: 2.5 rad/s
robot.rotation_velocity_limit.set(2.5)

# 肘部速度限制 [rad/s]
# 默认值: 2.175 rad/s (FR3推荐值)
# 最大值: 2.62 rad/s (Panda) / 2.175 rad/s (FR3)
robot.elbow_velocity_limit.set(2.175)

# 关节速度限制 [rad/s] (7个关节)
# 默认值 (Panda): [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
# 默认值 (FR3):   [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
robot.joint_velocity_limit.set([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

# ===== 加速度限制 (Acceleration Limits) =====

# 平移加速度限制 [m/s²]
# 默认值: 13.0 m/s² (FR3推荐值)
# 最大值: 9.0 m/s² (Panda) / 13.0 m/s² (FR3)
robot.translation_acceleration_limit.set(13.0)

# 旋转加速度限制 [rad/s²]
# 默认值: 25.0 rad/s² (FR3推荐值)
# 最大值: 17.0 rad/s² (Panda) / 25.0 rad/s² (FR3)
robot.rotation_acceleration_limit.set(25.0)

# 肘部加速度限制 [rad/s²]
# 默认值: 10.0 rad/s²
# 最大值: 10.0 rad/s²
robot.elbow_acceleration_limit.set(10.0)

# 关节加速度限制 [rad/s²] (7个关节)
# 默认值 (Panda): [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
# 默认值 (FR3):   [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
robot.joint_acceleration_limit.set([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

# ===== 加加速度限制 (Jerk Limits) =====

# 平移加加速度限制 [m/s³]
# 默认值: 6500.0 m/s³ (FR3推荐值)
# 最大值: 4500.0 m/s³ (Panda) / 6500.0 m/s³ (FR3)
robot.translation_jerk_limit.set(6500.0)

# 旋转加加速度限制 [rad/s³]
# 默认值: 12500.0 rad/s³ (FR3推荐值)
# 最大值: 8500.0 rad/s³ (Panda) / 12500.0 rad/s³ (FR3)
robot.rotation_jerk_limit.set(12500.0)

# 肘部加加速度限制 [rad/s³]
# 默认值: 5000.0 rad/s³
# 最大值: 5000.0 rad/s³
robot.elbow_jerk_limit.set(5000.0)

# 关节加加速度限制 [rad/s³] (7个关节)
# 默认值 (Panda): [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0]
# 默认值 (FR3):   [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]
robot.joint_jerk_limit.set([5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0])

# 获取每个限制的最大值（由Franka提供）
print(f"关节加加速度最大值: {robot.joint_jerk_limit.max}")
print(f"平移速度最大值: {robot.translation_velocity_limit.max}")
```

**类型说明:**
- `relative_dynamics_factor: float | RelativeDynamicsFactor` - 相对动力学因子
- 所有限制对象都有 `.set(value)` 和 `.max` 方法

#### 1.2.3 机器人模型属性

```python
# 获取机器人运动学模型
model: Model = robot.model

# 获取URDF模型字符串（需要libfranka >= 0.15.0）
urdf_str: str = robot.model_urdf
```

#### 1.2.4 机器人模型方法 (Model Methods)

`robot.model` 提供了一系列方法来访问机器人的运动学和动力学属性。这些对于高级控制算法非常有用。

```python
from franky import Robot, Frame

robot = Robot("10.90.90.1")
model = robot.model
state = robot.state

# 计算重力矢量 [Nm]
gravity_vector = model.gravity(state)

# 计算科里奥利和离心力矢量 [Nm]
coriolis_vector = model.coriolis(state)

# 获取质量矩阵 (7x7) [kg*m^2]
mass_matrix = model.mass(state)

# 获取特定坐标系下的位姿
pose_flange = model.pose(Frame.Flange, state)

# 获取特定坐标系下的雅可比矩阵 (6x7)
jacobian_ee = model.body_jacobian(Frame.EndEffector, state)
```

**主要方法:**
- `gravity(state, [gravity_earth])`: 计算重力矢量。
- `coriolis(state)`: 计算科里奥利力。
- `mass(state)`: 获取质量矩阵。
- `pose(frame, state)`: 获取指定 `Frame` （如 `Frame.Flange`, `Frame.Joint5` 等）的位姿。
- `body_jacobian(frame, state)`: 获取指定 `Frame` 的雅可比矩阵（相对于 `frame` 坐标系）。
- `zero_jacobian(frame, state)`: 获取指定 `Frame` 的雅可比矩阵（相对于基座坐标系）。

---

### 1.3 运动控制方法

#### 1.3.1 基本运动控制

```python
# 执行运动（阻塞式）
robot.move(motion)

# 执行运动（异步，非阻塞）
robot.move(motion, asynchronous=True)

# 执行运动时指定相对动力学因子
robot.move(motion, relative_dynamics_factor=0.8)

# 等待异步运动完成
robot.join_motion()
```

**参数:**
- `motion: Motion` - 运动对象（见第2部分详细说明）
- `asynchronous: bool` (可选, 默认False) - 是否异步执行
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选) - 此次运动的动力学因子

**返回值:**
- `bool` - 运动是否成功完成

---

### 1.4 阻抗控制方法

#### 1.4.1 关节阻抗

```python
# 设置关节阻抗 (7个关节)
K_theta = [600, 600, 600, 600, 250, 150, 50]  # [Nm/rad]
robot.set_joint_impedance(K_theta)
```

**参数:**
- `K_theta: List[float]` - 7个关节的刚度值 [Nm/rad]

**Franka默认值（参考）:**
- 关节刚度范围: 每个关节可在 0 到约 3000 Nm/rad 之间设置
- 常用软刚度: `[600, 600, 600, 600, 250, 150, 50]`
- 常用硬刚度: `[3000, 3000, 3000, 2500, 2500, 2000, 2000]`

**重要：关于Damping（阻尼）**
- ⚠️ **无法单独设置damping系数**
- Franka自动使用临界阻尼公式：`D = 2 × √K`
- 例如：K = 600 Nm/rad → D ≈ 48.99 Nm·s/rad
- 详见文档：`IMPEDANCE_CONTROL_DAMPING.md`

#### 1.4.2 笛卡尔阻抗

```python
# 设置笛卡尔阻抗
# K_x: [平移x, 平移y, 平移z, 旋转x, 旋转y, 旋转z]
K_x = [3000, 3000, 3000, 300, 300, 300]  # [N/m 和 Nm/rad]
robot.set_cartesian_impedance(K_x)
```

**参数:**
- `K_x: List[float]` - 6个笛卡尔方向的刚度值
  - 前3个: 平移刚度 [N/m]
  - 后3个: 旋转刚度 [Nm/rad]

**Franka默认值（参考）:**
- 平移刚度范围: 0 到约 5000 N/m
- 旋转刚度范围: 0 到约 300 Nm/rad
- 常用默认值: `[3000, 3000, 3000, 300, 300, 300]`
- 柔顺Z轴（装配任务）: `[3000, 3000, 500, 300, 300, 300]`

**关于Damping:**
- 笛卡尔阻尼同样自动计算：`D = 2 × √K`
- 例如：K_x = 3000 N/m → D_x ≈ 109.54 N·s/m

---

### 1.5 碰撞和接触配置

#### 1.5.1 碰撞行为设置

```python
# 设置碰撞检测阈值（完整版本）
robot.set_collision_behavior(
    lower_torque_thresholds_acceleration=[20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
    upper_torque_thresholds_acceleration=[20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
    lower_torque_thresholds_nominal=[20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
    upper_torque_thresholds_nominal=[20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
    lower_force_thresholds_acceleration=[20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
    upper_force_thresholds_acceleration=[20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
    lower_force_thresholds_nominal=[20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
    upper_force_thresholds_nominal=[20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
)
```

**参数说明:**
- `lower_torque_thresholds_acceleration: List[float]` - 关节力矩下限（加速时）[Nm]
- `upper_torque_thresholds_acceleration: List[float]` - 关节力矩上限（加速时）[Nm]
- `lower_torque_thresholds_nominal: List[float]` - 关节力矩下限（匀速时）[Nm]
- `upper_torque_thresholds_nominal: List[float]` - 关节力矩上限（匀速时）[Nm]
- `lower_force_thresholds_acceleration: List[float]` - 笛卡尔力下限（加速时）[N, N, N, Nm, Nm, Nm]
- `upper_force_thresholds_acceleration: List[float]` - 笛卡尔力上限（加速时）[N, N, N, Nm, Nm, Nm]
- `lower_force_thresholds_nominal: List[float]` - 笛卡尔力下限（匀速时）[N, N, N, Nm, Nm, Nm]
- `upper_force_thresholds_nominal: List[float]` - 笛卡尔力上限（匀速时）[N, N, N, Nm, Nm, Nm]

**franky默认值（在Robot初始化时自动设置）:**
```python
# 来自 Robot::Params
default_torque_threshold = 20.0  # Nm（应用于所有7个关节）
default_force_threshold = 30.0   # N 或 Nm（应用于所有6个笛卡尔方向）

# 实际设置为：
# - 关节力矩阈值: [20.0] * 7
# - 笛卡尔力/力矩阈值: [30.0] * 6
```

**推荐值参考:**
- **更敏感**（更容易触发碰撞检测）: 力矩 15-18 Nm, 力 15-20 N
- **默认**（平衡安全与稳定）: 力矩 20 Nm, 力 30 N  
- **更宽松**（减少误触发）: 力矩 25-30 Nm, 力 40-50 N

#### 1.5.2 简化版接触检测设置

```python
# 设置所有阈值相同（简化版）
robot.set_collision_behavior(
    [20.0] * 7,  # 关节力矩阈值 [Nm]
    [30.0, 30.0, 30.0, 40.0, 40.0, 40.0]  # 笛卡尔力/力矩阈值 [N, Nm]
)

# 或者使用标量（会自动扩展到所有关节/方向）
robot.set_collision_behavior(
    20.0,  # 所有关节使用相同的力矩阈值
    30.0   # 所有笛卡尔方向使用相同的力阈值
)
```

---

### 1.6 负载配置

```python
# 设置末端执行器负载
mass = 0.5  # kg
center_of_mass = [0.0, 0.0, 0.05]  # [m] 相对于法兰坐标系
inertia = [0.01, 0.0, 0.0, 0.01, 0.0, 0.01]  # 惯性张量 [kg·m²]

robot.set_load(mass, center_of_mass, inertia)

# 或只设置质量和质心（不设置惯性张量）
robot.set_load(mass, center_of_mass)
```

**参数:**
- `mass: float` - 负载质量 [kg]
  - 范围: 0.0 到约 3.0 kg（取决于机器人型号）
  - 默认: 0.0 kg（无负载）
- `center_of_mass: List[float]` - 质心位置 [x, y, z] [m]（相对于法兰坐标系）
  - 默认: `[0.0, 0.0, 0.0]`
- `load_inertia: List[float]` (可选) - 惯性张量 [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] [kg·m²]
  - 默认: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`（零惯性）
  - 对于对称负载，非对角线元素通常为0

**示例场景:**
```python
# 场景1: 轻型夹爪
robot.set_load(0.3, [0.0, 0.0, 0.04])  # 300g, 质心在法兰上方4cm

# 场景2: 重型工具
robot.set_load(
    mass=1.5,  
    center_of_mass=[0.0, 0.0, 0.1],
    inertia=[0.05, 0.0, 0.0, 0.05, 0.0, 0.02]  # 对称工具
)

# 场景3: 无负载（重置为默认值）
robot.set_load(0.0, [0.0, 0.0, 0.0])
```

---

### 1.7 EE（末端执行器）配置

```python
# 设置末端执行器相对于法兰的变换
NE_T_EE = Affine([0.0, 0.0, 0.1])  # 沿z轴偏移10cm
robot.set_EE(NE_T_EE)
```

**参数:**
- `NE_T_EE: Affine` - 从标称末端执行器(NE)到末端执行器(EE)的变换矩阵

---

### 1.8 错误恢复

```python
# 从错误状态恢复
robot.recover_from_errors()

# 检查是否有错误
has_errors = robot.has_errors()

# 读取一次机器人状态（在错误恢复前可能需要）
robot.read_once()
```

**返回值:**
- `has_errors() -> bool` - 机器人是否处于错误状态

---

### 1.9 其他方法

```python
# 停止机器人（急停）
robot.stop()

# 自动停止（根据当前控制模式选择合适的停止方式）
robot.automatic_error_recovery()

# 开启/关闭手动拖动示教 (Guiding Mode)
# guiding_mode: 6个布尔值，对应 [x, y, z, Rx, Ry, Rz] 方向是否允许拖动
# elbow: 布尔值，是否允许肘部拖动
robot.set_guiding_mode(
    guiding_mode=[True, True, True, False, False, False], # 只允许平移
    elbow=True
)

# 关闭拖动示教
robot.set_guiding_mode([False]*6, False)
```

---
<a name="part2"></a>
## 第二部分: Motion 类

franky支持4种控制模式，每种模式都有对应的运动类：
1. **关节位置控制** (Joint Position Control)
2. **关节速度控制** (Joint Velocity Control)
3. **笛卡尔位置控制** (Cartesian Position Control)
4. **笛卡尔速度控制** (Cartesian Velocity Control)

---

### 2.1 关节位置控制 (Joint Position Control)

#### 2.1.1 JointMotion - 点到点关节运动

```python
from franky import JointMotion

# 移动到指定关节角度（绝对位置）
target_joints = [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]  # [rad]
motion = JointMotion(target_joints)

# 可选参数
motion = JointMotion(
    target=target_joints,
    relative_dynamics_factor=0.5  # 使用50%的动力学限制
)
```

**参数:**
- `target: List[float]` - 目标关节角度 (7个值) [rad]
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选) - 相对动力学因子

---

#### 2.1.2 JointWaypoint - 关节路点

```python
from franky import JointWaypoint, JointState

# 方式1: 只指定位置（机器人会在此路点停止）
waypoint1 = JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])

# 方式2: 指定位置和速度（机器人会以指定速度通过此路点）
waypoint2 = JointWaypoint(
    JointState(
        position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
        velocity=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 非零速度意味着不停止
    )
)

# 方式3: 带相对动力学因子
waypoint3 = JointWaypoint(
    target=[0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9],
    relative_dynamics_factor=0.3
)
```

**参数:**
- `target: List[float] | JointState` - 目标关节位置或关节状态
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.1.3 JointWaypointMotion - 多路点关节运动

```python
from franky import JointWaypointMotion, JointWaypoint, JointState

# 创建多个路点的运动
motion = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),  # 停止
    JointWaypoint(
        JointState(
            position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
            velocity=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 不停止，以此速度通过
        )
    ),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])  # 最后停止
])
```

**参数:**
- `waypoints: List[JointWaypoint]` - 关节路点列表

---

#### 2.1.4 JointStopMotion - 停止运动（关节模式）

```python
from franky import JointStopMotion

# 在关节位置控制模式下停止机器人
motion = JointStopMotion()
```

**说明:** 用于在关节位置控制模式下平滑停止机器人

---

### 2.2 关节速度控制 (Joint Velocity Control)

#### 2.2.1 JointVelocityMotion - 恒定关节速度运动

```python
from franky import JointVelocityMotion, Duration

# 以指定速度运动一段时间
target_velocity = [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4]  # [rad/s]
motion = JointVelocityMotion(
    target=target_velocity,
    duration=Duration(1000)  # 持续1000毫秒
)

# 不指定持续时间（需要手动停止或通过反应停止）
motion = JointVelocityMotion(target_velocity)
```

**参数:**
- `target: List[float]` - 目标关节速度 (7个值) [rad/s]
- `duration: Duration` (可选) - 持续时间
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.2.2 JointVelocityWaypoint - 关节速度路点

```python
from franky import JointVelocityWaypoint, Duration

# 加速到指定速度并保持一段时间
waypoint = JointVelocityWaypoint(
    target=[0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4],
    hold_target_duration=Duration(1000)  # 保持目标速度1秒
)

# 不保持（立即过渡到下一个路点）
waypoint = JointVelocityWaypoint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
```

**参数:**
- `target: List[float]` - 目标关节速度 (7个值) [rad/s]
- `hold_target_duration: Duration` (可选) - 保持目标速度的时间
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.2.3 JointVelocityWaypointMotion - 多路点关节速度运动

```python
from franky import JointVelocityWaypointMotion, JointVelocityWaypoint, Duration

# 多段速度运动
motion = JointVelocityWaypointMotion([
    JointVelocityWaypoint(
        [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4],
        hold_target_duration=Duration(1000)  # 保持1秒
    ),
    JointVelocityWaypoint(
        [-0.1, -0.3, 0.1, 0.0, -0.1, 0.2, -0.4],
        hold_target_duration=Duration(2000)  # 保持2秒
    ),
    JointVelocityWaypoint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 停止
])
```

**参数:**
- `waypoints: List[JointVelocityWaypoint]` - 关节速度路点列表

**重要提示:** 必须在最后添加一个零速度路点来停止机器人！

---

#### 2.2.4 JointVelocityStopMotion - 停止运动（关节速度模式）

```python
from franky import JointVelocityStopMotion

# 在关节速度控制模式下停止机器人
motion = JointVelocityStopMotion()
```

---

### 2.3 笛卡尔位置控制 (Cartesian Position Control)

#### 2.3.1 CartesianMotion - 点到点笛卡尔运动

```python
from franky import CartesianMotion, Affine, RobotPose, ElbowState, ReferenceType
from scipy.spatial.transform import Rotation
import math

# 方式1: 移动到绝对位姿（只指定末端执行器位姿）
target_pose = Affine([0.4, -0.2, 0.3])  # 位置 [m]
motion = CartesianMotion(target_pose)

# 方式2: 指定位姿和旋转
quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
target_pose = Affine([0.4, -0.2, 0.3], quat)
motion = CartesianMotion(target_pose)

# 方式3: 指定末端执行器位姿和肘部位置
robot_pose = RobotPose(
    end_effector_pose=Affine([0.4, -0.2, 0.3], quat),
    elbow_state=ElbowState(0.3)  # 肘部角度 [rad]
)
motion = CartesianMotion(robot_pose)

# 方式4: 相对运动（相对于当前位姿）
motion = CartesianMotion(
    Affine([0.2, 0.0, 0.0]),  # 沿当前X轴移动20cm
    ReferenceType.Relative
)

# 方式5: 带相对动力学因子
motion = CartesianMotion(
    target_pose,
    relative_dynamics_factor=0.5
)
```

**参数:**
- `target: Affine | RobotPose` - 目标位姿或机器人位姿
- `reference_type: ReferenceType` (可选, 默认Absolute) - 参考类型
  - `ReferenceType.Absolute` - 绝对位姿
  - `ReferenceType.Relative` - 相对于当前位姿
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.3.2 CartesianWaypoint - 笛卡尔路点

```python
from franky import CartesianWaypoint, CartesianState, Affine, RobotPose, Twist, ReferenceType

# 方式1: 只指定位姿（机器人会停止）
waypoint1 = CartesianWaypoint(Affine([0.5, -0.2, 0.3]))

# 方式2: 指定位姿和速度（机器人以指定速度通过）
waypoint2 = CartesianWaypoint(
    CartesianState(
        pose=Affine([0.4, -0.1, 0.3]),
        velocity=Twist([-0.01, 0.01, 0.0])  # 线速度 [m/s]
    )
)

# 方式3: 相对路点（相对于上一个路点）
waypoint3 = CartesianWaypoint(
    Affine([0.2, 0.0, 0.0]),
    ReferenceType.Relative
)

# 方式4: 带动力学因子（此路点使用50%速度）
waypoint4 = CartesianWaypoint(
    Affine([0.3, 0.0, 0.3]),
    relative_dynamics_factor=RelativeDynamicsFactor(0.5, 1.0, 1.0)
)
```

**参数:**
- `target: Affine | RobotPose | CartesianState` - 目标位姿或笛卡尔状态
- `reference_type: ReferenceType` (可选)
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.3.3 CartesianWaypointMotion - 多路点笛卡尔运动

```python
from franky import CartesianWaypointMotion, CartesianWaypoint, Affine, CartesianState, Twist

# 多路点运动
motion = CartesianWaypointMotion([
    CartesianWaypoint(Affine([0.5, -0.2, 0.3])),  # 停止
    CartesianWaypoint(
        CartesianState(
            pose=Affine([0.4, -0.1, 0.3]),
            velocity=Twist([-0.01, 0.01, 0.0])  # 不停止
        )
    ),
    CartesianWaypoint(Affine([0.3, 0.0, 0.3]))  # 最后停止
])
```

**参数:**
- `waypoints: List[CartesianWaypoint]` - 笛卡尔路点列表

---

#### 2.3.4 CartesianStopMotion - 停止运动（笛卡尔模式）

```python
from franky import CartesianStopMotion

# 在笛卡尔位置控制模式下停止机器人
motion = CartesianStopMotion()
```

---

### 2.4 笛卡尔速度控制 (Cartesian Velocity Control)

#### 2.4.1 CartesianVelocityMotion - 笛卡尔速度运动

```python
from franky import CartesianVelocityMotion, Twist, RobotVelocity, Duration

# 方式1: 只指定末端执行器速度
linear_vel = [0.2, -0.1, 0.1]  # [m/s]
angular_vel = [0.1, -0.1, 0.2]  # [rad/s]
twist = Twist(linear_vel, angular_vel)
motion = CartesianVelocityMotion(twist)

# 方式2: 带持续时间
motion = CartesianVelocityMotion(
    twist,
    duration=Duration(2000)  # 持续2秒
)

# 方式3: 同时指定末端执行器速度和肘部速度
robot_velocity = RobotVelocity(
    end_effector_twist=Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
    elbow_velocity=-0.2  # [rad/s]
)
motion = CartesianVelocityMotion(robot_velocity)
```

**参数:**
- `target: Twist | RobotVelocity` - 目标速度（末端执行器或完整机器人）
- `duration: Duration` (可选) - 持续时间
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.4.2 CartesianVelocityWaypoint - 笛卡尔速度路点

```python
from franky import CartesianVelocityWaypoint, Twist, Duration

# 加速到指定速度并保持
waypoint = CartesianVelocityWaypoint(
    target=Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
    hold_target_duration=Duration(1000)  # 保持1秒
)

# 立即过渡到下一个路点
waypoint = CartesianVelocityWaypoint(Twist())  # 零速度
```

**参数:**
- `target: Twist | RobotVelocity` - 目标速度
- `hold_target_duration: Duration` (可选) - 保持目标速度的时间
- `relative_dynamics_factor: float | RelativeDynamicsFactor` (可选)

---

#### 2.4.3 CartesianVelocityWaypointMotion - 多路点笛卡尔速度运动

```python
from franky import CartesianVelocityWaypointMotion, CartesianVelocityWaypoint, Twist, Duration

# 多段速度运动
motion = CartesianVelocityWaypointMotion([
    CartesianVelocityWaypoint(
        Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
        hold_target_duration=Duration(1000)
    ),
    CartesianVelocityWaypoint(
        Twist([-0.2, 0.1, -0.1], [-0.1, 0.1, -0.2]),
        hold_target_duration=Duration(2000)
    ),
    CartesianVelocityWaypoint(Twist())  # 停止
])
```

**参数:**
- `waypoints: List[CartesianVelocityWaypoint]` - 笛卡尔速度路点列表

**重要提示:** 必须在最后添加一个零速度路点来停止机器人！

---

#### 2.4.4 CartesianVelocityStopMotion - 停止运动（笛卡尔速度模式）

```python
from franky import CartesianVelocityStopMotion

# 在笛卡尔速度控制模式下停止机器人
motion = CartesianVelocityStopMotion()
```

---

### 2.5 运动回调 (Motion Callbacks)

所有运动类型都支持回调函数，回调会在每个控制周期（1kHz）被调用。

```python
from franky import RobotState, Duration, JointPositions

def motion_callback(
    robot_state: RobotState,
    time_step: Duration,
    rel_time: Duration,
    abs_time: Duration,
    control_signal  # 类型取决于运动类型
):
    """
    参数说明:
    - robot_state: 当前机器人状态
    - time_step: 时间步长
    - rel_time: 相对时间（运动开始后的时间）
    - abs_time: 绝对时间（总运动时间）
    - control_signal: 控制信号
      * JointMotion -> JointPositions
      * JointVelocityMotion -> JointVelocities
      * CartesianMotion -> CartesianPose
      * CartesianVelocityMotion -> CartesianVelocities
    """
    print(f"时间: {abs_time}, 控制信号: {control_signal}")

# 注册回调
motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
motion.register_callback(motion_callback)

# 执行运动
robot.move(motion)
```

**注意:** Python中的回调不在实时线程中执行，而是在单独的线程中，以避免阻塞控制线程。

---
<a name="part3"></a>
## 第三部分: 状态类与反应系统

本部分介绍机器人状态类和实时反应系统的Python API。

---

### 3.1 RobotState - 完整机器人状态

`RobotState` 继承自libfranka的 `franka::RobotState`，包含机器人的所有状态信息。

```python
from franky import Robot

robot = Robot("10.90.90.1")
state = robot.state  # 获取当前状态
```

#### 3.1.1 位姿相关字段

```python
# 末端执行器位姿（从基座到末端执行器）
O_T_EE: List[float] = state.O_T_EE  # 4x4变换矩阵（列优先，16个元素）

# 法兰位姿（从基座到法兰）
O_T_EE_c: List[float] = state.O_T_EE_c  # 4x4变换矩阵

# 法兰到末端执行器的变换
F_T_EE: List[float] = state.F_T_EE  # 4x4变换矩阵

# 标称末端执行器到末端执行器的变换
NE_T_EE: List[float] = state.NE_T_EE  # 4x4变换矩阵

# 末端执行器到夹爪的变换
EE_T_K: List[float] = state.EE_T_K  # 4x4变换矩阵
```

#### 3.1.2 关节状态

```python
# 关节位置 [rad]
q: List[float] = state.q  # 7个关节，测量值

# 关节位置（期望值）[rad]
q_d: List[float] = state.q_d  # 7个关节

# 关节速度 [rad/s]
dq: List[float] = state.dq  # 7个关节，测量值

# 关节速度（期望值）[rad/s]
dq_d: List[float] = state.dq_d  # 7个关节

# 关节加速度（期望值）[rad/s²]
ddq_d: List[float] = state.ddq_d  # 7个关节
```

#### 3.1.3 力矩和力

```python
# 测量的关节力矩 [Nm]
tau_J: List[float] = state.tau_J  # 7个关节

# 期望的关节力矩 [Nm]
tau_J_d: List[float] = state.tau_J_d  # 7个关节

# 外部力矩 [Nm]（在关节空间）
tau_ext_hat_filtered: List[float] = state.tau_ext_hat_filtered  # 7个关节

# 外部力/力矩 [N, Nm]（在K坐标系，即夹爪坐标系）
K_F_ext_hat_K: List[float] = state.K_F_ext_hat_K  # [Fx, Fy, Fz, Mx, My, Mz]

# 外部力/力矩 [N, Nm]（在O坐标系，即基座坐标系）
O_F_ext_hat_K: List[float] = state.O_F_ext_hat_K  # [Fx, Fy, Fz, Mx, My, Mz]
```

#### 3.1.4 末端执行器状态

```python
# 末端执行器位姿的导数（笛卡尔速度）
O_dP_EE_c: List[float] = state.O_dP_EE_c  # [dx, dy, dz, wx, wy, wz] [m/s, rad/s]

# 末端执行器位姿的二阶导数（笛卡尔加速度）
O_ddP_EE_c: List[float] = state.O_ddP_EE_c  # [ddx, ddy, ddz, dwx, dwy, dwz] [m/s², rad/s²]

# 肘部位置 [rad]（第4关节的位置）
elbow: List[float] = state.elbow  # [x, y] (2D投影)

# 肘部位置（期望值）[rad]
elbow_d: List[float] = state.elbow_d  # [x, y]

# 肘部速度 [rad/s]
delbow_c: List[float] = state.delbow_c  # [dx, dy]

# 肘部加速度 [rad/s²]
ddelbow_c: List[float] = state.ddelbow_c  # [ddx, ddy]
```

#### 3.1.5 雅可比矩阵

```python
# 末端执行器雅可比矩阵（从基座坐标系）
O_Jac_EE: List[float] = state.O_Jac_EE  # 6x7矩阵（列优先，42个元素）

# 末端执行器零空间雅可比矩阵
O_Jac_EE_zero: List[float] = state.O_Jac_EE_zero  # 7x7矩阵
```

#### 3.1.6 惯量和质量矩阵

```python
# 质量矩阵 [kg]
m_ee: float = state.m_ee  # 末端执行器质量

# 惯性矩阵的F_x_Cee部分 [m]
F_x_Cee: List[float] = state.F_x_Cee  # [x, y, z]

# 惯性矩阵的I_ee [kg*m²]
I_ee: List[float] = state.I_ee  # 9个元素（3x3矩阵）

# 总惯性矩阵 [kg*m²]
m_total: float = state.m_total  # 包含负载的总质量

# 质心位置 [m]
F_x_Ctotal: List[float] = state.F_x_Ctotal  # [x, y, z]

# 总惯性张量 [kg*m²]
I_total: List[float] = state.I_total  # 9个元素
```

#### 3.1.7 控制相关

```python
# 肘部命令值 [rad]（用户设置的期望肘部位置）
elbow_c: List[float] = state.elbow_c  # [x, y]

# 关节碰撞检测
joint_collision: List[float] = state.joint_collision  # 7个关节，0或1

# 关节接触检测
joint_contact: List[float] = state.joint_contact  # 7个关节，0或1

# 笛卡尔碰撞检测
cartesian_collision: List[float] = state.cartesian_collision  # [x, y, z, rx, ry, rz]

# 笛卡尔接触检测
cartesian_contact: List[float] = state.cartesian_contact  # [x, y, z, rx, ry, rz]
```

#### 3.1.8 机器人模式和时间

```python
# 当前控制器模式
control_command_success_rate: float = state.control_command_success_rate  # 0.0 到 1.0

# 机器人模式
robot_mode: RobotMode = state.robot_mode
# 可能的值:
# - RobotMode.Other
# - RobotMode.Idle
# - RobotMode.Move
# - RobotMode.Guiding
# - RobotMode.Reflex
# - RobotMode.UserStopped
# - RobotMode.AutomaticErrorRecovery

# 时间 [ms]
time: Duration = state.time  # 自控制器启动后的时间
```

#### 3.1.9 错误状态

```python
# 当前错误
current_errors: Errors = state.current_errors
# 包含的布尔字段:
# - joint_position_limits_violation
# - cartesian_position_limits_violation
# - self_collision_avoidance_violation
# - joint_velocity_violation
# - cartesian_velocity_violation
# - force_control_safety_violation
# - joint_reflex
# - cartesian_reflex
# - max_goal_pose_deviation_violation
# - max_path_pose_deviation_violation
# - cartesian_position_motion_generator_start_pose_invalid
# - joint_motion_generator_position_limits_violation
# - joint_motion_generator_velocity_limits_violation
# - joint_motion_generator_velocity_discontinuity
# - joint_motion_generator_acceleration_discontinuity
# - cartesian_position_motion_generator_elbow_limit_violation
# - ... (还有更多)

# 上一次错误
last_motion_errors: Errors = state.last_motion_errors
```

---

### 3.2 CartesianState - 笛卡尔状态

包含末端执行器的位姿和速度信息。

```python
from franky import Robot

robot = Robot("10.90.90.1")
cartesian_state = robot.current_cartesian_state
```

#### 字段说明

```python
# 机器人位姿（包含末端执行器位姿和肘部位置）
pose: RobotPose = cartesian_state.pose

# 末端执行器位姿
ee_pose: Affine = cartesian_state.pose.end_effector_pose

# 肘部状态
elbow: ElbowState = cartesian_state.pose.elbow_state
elbow_pos: float = elbow.position  # [rad] 或 None

# 机器人速度（包含末端执行器速度和肘部速度）
velocity: RobotVelocity = cartesian_state.velocity

# 末端执行器扭转速度（线速度和角速度）
ee_twist: Twist = cartesian_state.velocity.end_effector_twist
linear_vel = ee_twist.linear  # [vx, vy, vz] [m/s]
angular_vel = ee_twist.angular  # [wx, wy, wz] [rad/s]

# 肘部速度
elbow_vel: float = cartesian_state.velocity.elbow_velocity  # [rad/s] 或 None
```

---

### 3.3 JointState - 关节状态

包含所有关节的位置和速度信息。

```python
from franky import Robot

robot = Robot("10.90.90.1")
joint_state = robot.current_joint_state
```

#### 字段说明

```python
# 关节位置 [rad]
position: List[float] = joint_state.position  # 7个关节

# 关节速度 [rad/s]
velocity: List[float] = joint_state.velocity  # 7个关节
```

---

### 3.4 辅助类

#### 3.4.1 RobotPose - 机器人位姿

```python
from franky import RobotPose, Affine, ElbowState

# 只指定末端执行器位姿
pose1 = RobotPose(Affine([0.4, -0.2, 0.3]))

# 指定末端执行器位姿和肘部位置
pose2 = RobotPose(
    end_effector_pose=Affine([0.4, -0.2, 0.3]),
    elbow_state=ElbowState(0.3)  # [rad]
)
```

**字段:**
- `end_effector_pose: Affine` - 末端执行器位姿
- `elbow_state: ElbowState | None` - 肘部状态（可选）

#### 3.4.2 RobotVelocity - 机器人速度

```python
from franky import RobotVelocity, Twist

# 只指定末端执行器速度
vel1 = RobotVelocity(Twist([0.1, 0.0, 0.0]))

# 指定末端执行器速度和肘部速度
vel2 = RobotVelocity(
    end_effector_twist=Twist([0.1, 0.0, 0.0], [0.0, 0.0, 0.1]),
    elbow_velocity=-0.2  # [rad/s]
)
```

**字段:**
- `end_effector_twist: Twist` - 末端执行器扭转速度
- `elbow_velocity: float | None` - 肘部速度（可选）

#### 3.4.3 Twist - 扭转速度

```python
from franky import Twist

# 只指定线速度
twist1 = Twist([0.1, 0.0, 0.0])  # [m/s]

# 指定线速度和角速度
twist2 = Twist(
    linear=[0.1, 0.0, 0.0],   # [m/s]
    angular=[0.0, 0.0, 0.1]   # [rad/s]
)

# 零速度
twist_zero = Twist()
```

**字段:**
- `linear: List[float]` - 线速度 [vx, vy, vz] [m/s]
- `angular: List[float]` - 角速度 [wx, wy, wz] [rad/s]

#### 3.4.4 ElbowState - 肘部状态

```python
from franky import ElbowState

# 创建肘部状态
elbow = ElbowState(0.3)  # [rad]

# 访问位置
position = elbow.position  # float [rad]
```

---

### 6. Gripper - 夹爪控制

`Gripper` 类用于控制Franka的官方夹爪。

#### 6.1 Gripper 初始化

```python
from franky import Gripper

# 连接到夹爪（使用与机器人相同的IP地址）
gripper = Gripper("10.90.90.1")
```

#### 6.2 Gripper 方法

```python
# 1. 初始化夹爪（回零）
# 这是一个阻塞操作，会等待夹爪完成回零。
success = gripper.homing()

# 2. 张开夹爪
# speed: 张开速度 [m/s] (最大约 0.1 m/s)
gripper.open(speed=0.1)

# 3. 移动到指定宽度
# width: 目标宽度 [m] (0.0 到 gripper.max_width)
# speed: 移动速度 [m/s]
gripper.move(width=0.04, speed=0.05)

# 4. 夹取物体
# width: 尝试夹取时，夹爪应移动到的最小宽度 [m]
# speed: 夹取速度 [m/s]
# force: 夹取力 [N] (1 到 70 N)
# epsilon_inner, epsilon_outer: 宽度容差，用于判断是否成功夹取
is_grasped = gripper.grasp(width=0.0, speed=0.05, force=40.0)
if is_grasped:
    print("成功夹取物体！")

# 5. 停止夹爪运动
gripper.stop()

# 所有方法都有异步版本，例如:
future = gripper.homing_async()
# ... 做其他事 ...
success = future.get() # 等待结果
```

**主要方法:**
- `homing() -> bool`: 执行回零操作。必须在任何其他操作之前调用。
- `open(speed) -> bool`: 完全张开夹爪。
- `move(width, speed) -> bool`: 移动到指定宽度。
- `grasp(width, speed, force, [epsilon_inner], [epsilon_outer]) -> bool`: 尝试夹取物体。如果夹爪宽度在 `width ± epsilon` 范围内停止，则认为夹取成功。
- `stop() -> bool`: 立即停止夹爪运动。
- 所有方法都有一个 `_async` 的异步版本（如 `homing_async`），它会立即返回一个 `BoolFuture` 对象，你可以稍后通过 `.get()` 获取结果。

#### 6.3 Gripper 属性

```python
from franky import Gripper

gripper = Gripper("10.90.90.1")

# 异步移动
future = gripper.move_async(0.05, speed=0.02)

# 做其他事情...
print("夹爪正在移动...")

# 等待完成（最多等待2秒）
if future.wait(2.0):
    success = future.get()
    print(f"移动{'成功' if success else '失败'}")
else:
    # 超时，停止夹爪
    gripper.stop()
    future.wait()  # 等待停止完成
    print("移动超时，已停止")

# # 获取夹爪当前状态
# state = gripper.state

# # 当前夹爪宽度 [m]
# current_width = gripper.width
# # 或者 state.width

# # 夹爪是否夹住物体
# is_grasped = gripper.is_grasped
# # 或者 state.is_grasped

# # 夹爪最大宽度 [m]
# max_width = gripper.max_width
# # 或者 state.max_width
```

---

<a name="appendix"></a>
## 附录: 默认参数速查表

本附录提供了franky API中常用参数的默认值和推荐范围，方便快速查阅。

---

### 1. 动力学限制默认值

#### 1.1 Franka Panda（系统版本 ≤ 4.x）

```python
from franky import Robot

robot = Robot("10.90.90.1")

# ===== 速度限制默认值 =====
# 平移速度: 默认 1.7 m/s, 最大 3.0 m/s
robot.translation_velocity_limit.set(1.7)

# 旋转速度: 默认 2.5 rad/s, 最大 2.5 rad/s
robot.rotation_velocity_limit.set(2.5)

# 肘部速度: 默认 2.175 rad/s, 最大 2.62 rad/s
robot.elbow_velocity_limit.set(2.175)

# 关节速度: 默认值 [rad/s]
robot.joint_velocity_limit.set([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

# ===== 加速度限制默认值 =====
# 平移加速度: 默认 13.0 m/s², 最大 9.0 m/s²
robot.translation_acceleration_limit.set(13.0)

# 旋转加速度: 默认 25.0 rad/s², 最大 17.0 rad/s²
robot.rotation_acceleration_limit.set(25.0)

# 肘部加速度: 默认 10.0 rad/s², 最大 10.0 rad/s²
robot.elbow_acceleration_limit.set(10.0)

# 关节加速度: 默认值 [rad/s²]
robot.joint_acceleration_limit.set([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

# ===== 加加速度限制默认值 =====
# 平移加加速度: 默认 6500.0 m/s³, 最大 4500.0 m/s³
robot.translation_jerk_limit.set(6500.0)

# 旋转加加速度: 默认 12500.0 rad/s³, 最大 8500.0 rad/s³
robot.rotation_jerk_limit.set(12500.0)

# 肘部加加速度: 默认 5000.0 rad/s³, 最大 5000.0 rad/s³
robot.elbow_jerk_limit.set(5000.0)

# 关节加加速度: 默认值 [rad/s³]
robot.joint_jerk_limit.set([5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0])
```

#### 1.2 Franka Research 3 (FR3)（系统版本 ≥ 5.x）

```python
# ===== 速度限制默认值 (FR3) =====
robot.translation_velocity_limit.set(1.7)  # m/s (最大 1.7)
robot.rotation_velocity_limit.set(2.5)    # rad/s (最大 2.5)
robot.elbow_velocity_limit.set(2.175)     # rad/s (最大 2.175)
robot.joint_velocity_limit.set([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])  # rad/s

# ===== 加速度限制默认值 (FR3) =====
robot.translation_acceleration_limit.set(13.0)  # m/s² (最大 13.0)
robot.rotation_acceleration_limit.set(25.0)    # rad/s² (最大 25.0)
robot.elbow_acceleration_limit.set(10.0)       # rad/s² (最大 10.0)
robot.joint_acceleration_limit.set([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])  # rad/s²

# ===== 加加速度限制默认值 (FR3) =====
robot.translation_jerk_limit.set(6500.0)   # m/s³ (最大 6500.0)
robot.rotation_jerk_limit.set(12500.0)    # rad/s³ (最大 12500.0)
robot.elbow_jerk_limit.set(5000.0)        # rad/s³ (最大 5000.0)
robot.joint_jerk_limit.set([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])  # rad/s³
```

#### 1.3 相对动力学因子

```python
# 全局相对动力学因子
# 默认值: 1.0（100%，使用全部限制）
robot.relative_dynamics_factor = 1.0

# 或分别设置
from franky import RelativeDynamicsFactor
robot.relative_dynamics_factor = RelativeDynamicsFactor(
    velocity=1.0,      # 100% 速度
    acceleration=1.0,  # 100% 加速度
    jerk=1.0          # 100% 加加速度
)

# 推荐的安全起始值
robot.relative_dynamics_factor = 0.05  # 5% 的限制（非常慢，很安全）
```

---

### 2. 阻抗控制默认值

#### 2.1 关节阻抗

```python
# 关节刚度 [Nm/rad]
# Franka没有固定的"默认值"，但以下是常用值：

# 柔软刚度（推荐用于接触任务）
soft_stiffness = [600, 600, 600, 600, 250, 150, 50]
robot.set_joint_impedance(soft_stiffness)

# 中等刚度（平衡性能和柔顺性）
medium_stiffness = [1500, 1500, 1500, 1000, 600, 400, 200]
robot.set_joint_impedance(medium_stiffness)

# 硬刚度（高精度定位）
stiff_stiffness = [3000, 3000, 3000, 2500, 2500, 2000, 2000]
robot.set_joint_impedance(stiff_stiffness)

# 刚度范围：每个关节约 0 到 3000 Nm/rad
```

**关于阻尼（Damping）:**
```python
# ⚠️ 无法单独设置阻尼！
# Franka自动计算: D = 2 × √K (临界阻尼)

# 示例计算:
import math
K_theta = [600, 600, 600, 600, 250, 150, 50]
D_theta = [2 * math.sqrt(k) for k in K_theta]
# 结果: [48.99, 48.99, 48.99, 48.99, 31.62, 24.49, 14.14] Nm·s/rad
```

#### 2.2 笛卡尔阻抗

```python
# 笛卡尔刚度 [N/m, N/m, N/m, Nm/rad, Nm/rad, Nm/rad]
# 格式: [平移x, 平移y, 平移z, 旋转x, 旋转y, 旋转z]

# 默认/标准刚度
default_cartesian = [3000, 3000, 3000, 300, 300, 300]
robot.set_cartesian_impedance(default_cartesian)

# 柔顺Z轴（装配任务常用）
soft_z = [3000, 3000, 500, 300, 300, 300]
robot.set_cartesian_impedance(soft_z)

# 高刚度（精密操作）
high_stiffness = [5000, 5000, 5000, 300, 300, 300]
robot.set_cartesian_impedance(high_stiffness)

# 刚度范围：
# - 平移: 约 0 到 5000 N/m
# - 旋转: 约 0 到 300 Nm/rad
```

**笛卡尔阻尼计算:**
```python
import math
K_x = [3000, 3000, 3000, 300, 300, 300]
D_x = [2 * math.sqrt(k) for k in K_x]
# 结果: [109.54, 109.54, 109.54, 34.64, 34.64, 34.64]
# 单位: [N·s/m, N·s/m, N·s/m, Nm·s/rad, Nm·s/rad, Nm·s/rad]
```

---

### 3. 碰撞检测默认值

#### 3.1 Robot初始化时的默认值

```python
# 在 Robot::Params 中定义的默认值
default_torque_threshold = 20.0  # Nm
default_force_threshold = 30.0   # N 或 Nm

# Robot初始化时会自动调用:
# robot.set_collision_behavior(
#     [20.0] * 7,  # 所有关节力矩阈值
#     [30.0] * 6   # 所有笛卡尔力/力矩阈值
# )
```

#### 3.2 推荐的碰撞阈值

```python
# 更敏感（适用于人机协作）
sensitive_collision = {
    'torque': [15.0, 15.0, 15.0, 15.0, 12.0, 10.0, 8.0],  # Nm
    'force': [15.0, 15.0, 15.0, 20.0, 20.0, 20.0]         # N, Nm
}
robot.set_collision_behavior(
    sensitive_collision['torque'],
    sensitive_collision['force']
)

# 标准（默认值，平衡性）
standard_collision = {
    'torque': [20.0] * 7,
    'force': [30.0, 30.0, 30.0, 40.0, 40.0, 40.0]
}
robot.set_collision_behavior(
    standard_collision['torque'],
    standard_collision['force']
)

# 宽松（减少误触发）
loose_collision = {
    'torque': [25.0, 25.0, 25.0, 25.0, 20.0, 18.0, 15.0],
    'force': [40.0, 40.0, 40.0, 50.0, 50.0, 50.0]
}
robot.set_collision_behavior(
    loose_collision['torque'],
    loose_collision['force']
)
```

---

### 4. 负载配置默认值

```python
# 默认值（无负载）
default_load = {
    'mass': 0.0,  # kg
    'center_of_mass': [0.0, 0.0, 0.0],  # m (相对于法兰)
    'inertia': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # kg·m² [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
}
robot.set_load(0.0, [0.0, 0.0, 0.0])

# 常见负载示例
# 轻型夹爪
robot.set_load(0.3, [0.0, 0.0, 0.04])  # 300g, 质心在法兰上方4cm

# 标准夹爪
robot.set_load(0.7, [0.0, 0.0, 0.06], [0.01, 0.0, 0.0, 0.01, 0.0, 0.01])

# 重型工具
robot.set_load(1.5, [0.0, 0.0, 0.1], [0.05, 0.0, 0.0, 0.05, 0.0, 0.02])

# 负载范围: 
# - 质量: 0 到约 3.0 kg（取决于机器人型号）
# - 质心: 通常在法兰上方 0-20cm
```

---

### 5. Robot初始化参数

```python
from franky import Robot, RealtimeConfig

# 默认初始化（使用所有默认参数）
robot = Robot("10.90.90.1")

# 完整的参数结构（Robot::Params）
class RobotParams:
    """Robot的默认参数（来自C++源码）"""
    
    # 动力学因子
    relative_dynamics_factor = 1.0  # 100%
    
    # 碰撞检测默认值
    default_torque_threshold = 20.0  # Nm
    default_force_threshold = 30.0   # N 或 Nm
    
    # 实时配置
    # realtime_config = RealtimeConfig.kEnforce  # 强制实时模式
    
    # Kalman滤波器参数（高级用户使用，通常保持默认）
    kalman_q_process_var = 0.0001
    kalman_dq_process_var = 0.001
    kalman_ddq_process_var = 0.1
    kalman_control_process_var = 1
    kalman_q_obs_var = 0.01
    kalman_dq_obs_var = 0.1
    kalman_q_d_obs_var = 0.0001
    kalman_dq_d_obs_var = 0.0001
    kalman_ddq_d_obs_var = 0.0001
    kalman_control_adaptation_rate = 0.1

# 使用自定义实时配置
robot = Robot("10.90.90.1", realtime_config=RealtimeConfig.IgnoreRealtime)
```

---

### 6. 常用配置组合

#### 6.1 新手安全配置

```python
from franky import Robot

robot = Robot("10.90.90.1")

# 1. 设置很低的动力学因子（慢速）
robot.relative_dynamics_factor = 0.05  # 5% 速度

# 2. 使用柔软的阻抗
robot.set_joint_impedance([600, 600, 600, 600, 250, 150, 50])
robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])

# 3. 使用敏感的碰撞检测
robot.set_collision_behavior([15.0] * 7, [20.0] * 6)

# 4. 设置实际负载
robot.set_load(0.5, [0.0, 0.0, 0.05])  # 根据实际情况调整
```

#### 6.2 生产环境配置

```python
robot = Robot("10.90.90.1")

# 1. 适中的速度
robot.relative_dynamics_factor = 0.3  # 30% 速度

# 2. 中等刚度（平衡性能）
robot.set_joint_impedance([1500, 1500, 1500, 1000, 600, 400, 200])
robot.set_cartesian_impedance([3500, 3500, 3500, 300, 300, 300])

# 3. 标准碰撞检测
robot.set_collision_behavior([20.0] * 7, [30.0] * 6)

# 4. 精确负载配置
robot.set_load(0.8, [0.0, 0.0, 0.06], [0.015, 0.0, 0.0, 0.015, 0.0, 0.012])
```
