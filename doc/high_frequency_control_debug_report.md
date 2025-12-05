# Franky 高频遥操作控制调试报告

> 日期：2025-12-02  
> 作者：调试过程记录  
> 目标：实现 20-30Hz 的高频遥操作控制

---

## 1. 问题背景

### 1.1 原始需求

使用 SpaceMouse 遥操作 Franka Panda 机器人进行数据采集，希望达到 **20-30Hz** 的控制频率。

### 1.2 原始实现

```python
# data_collection_with_ik.py - 单进程架构
while True:
    control = space_mouse.control           # 读取 SpaceMouse
    target_joints = compute_ik(control)     # IK 计算
    robot.move(motion, asynchronous=True)   # 发送命令
    time.sleep(control_period)              # 等待下一周期
```

### 1.3 观察到的问题

| 设置频率 | 实际频率 | 问题 |
|---------|---------|------|
| 15 Hz   | ~10 Hz  | 无法达到目标频率 |
| 20 Hz   | ~10 Hz  | 更差 |

---

## 2. 核心发现：为什么单进程无法达到高频率？

### 2.1 诊断过程

添加详细计时统计后发现：

```
[Sleep Analysis]
  Requested: mean=55.23ms
  Actual:    mean=88.45ms
  Overshoot: mean=33.22ms, max=84.12ms  ← 异常！
```

**`time.sleep()` 请求 55ms，实际睡了 88ms，多出 33ms！**

### 2.2 尝试 Busy Wait

```python
# 用忙等待代替 sleep
while time.time() - start < target_time:
    pass
```

结果：**仍然有 33ms 的时间跳变！**

这说明问题不在 `time.sleep()`，而是 **Python 主线程本身被挂起了**。

### 2.3 根本原因：franky 实时线程抢占

franky 库的 C++ 底层使用 **SCHED_FIFO** 实时调度策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    Linux 进程调度                            │
├─────────────────────────────────────────────────────────────┤
│  SCHED_FIFO (实时)     │  SCHED_OTHER (普通)                 │
│  ├─ franky 控制线程    │  ├─ Python 主线程                   │
│  │  优先级: 高         │  │  优先级: 普通                     │
│  │  1kHz 控制循环      │  │  用户代码                         │
│  └─ 会抢占普通线程     │  └─ 被抢占时挂起                     │
└─────────────────────────────────────────────────────────────┘
```

**时间线分析：**

```
时间 →
Python: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░░
franky:     ████████████████████████████████            ████████████
            ↑                              ↑            ↑
            Python被抢占                    恢复执行     再次被抢占
            
抢占持续时间: 25-84ms (取决于 franky 内部处理)
```

---

## 3. 解决方案：多进程架构

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     主进程 (Python)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ SpaceMouse  │→ │  IK 计算    │→ │ command_queue.put() │  │
│  │ 读取        │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └──────────┬──────────┘  │
│                                                │             │
│  不受 franky 实时线程影响！                      │             │
└────────────────────────────────────────────────┼─────────────┘
                                                 │
                    multiprocessing.Queue        │
                                                 ↓
┌─────────────────────────────────────────────────────────────┐
│                   控制进程 (独立 Python 进程)                 │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │ command_queue.get() │→ │ robot.move(motion, async)   │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
│                                                              │
│  这个进程会被 franky 实时线程抢占，但不影响主进程！            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 为什么多进程可以达到高频率？

**关键洞察：franky 的实时线程只能抢占它所在的进程！**

| 架构 | 主进程被抢占？ | 能达到目标频率？ |
|------|---------------|-----------------|
| 单进程 | ✅ 是，被 franky 线程抢占 | ❌ 否，~10Hz |
| 多进程 | ❌ 否，robot 在另一个进程 | ✅ 是，30Hz |

```python
# test_multiprocess_control.py - 验证结果
Target frequency: 30.0 Hz
Actual frequency: 29.96 Hz  ← 成功！
Loop time: mean=33.37ms, std=0.15ms
```

### 3.3 代码实现

```python
from multiprocessing import Process, Queue, Value

def control_process(robot_ip, command_queue, running):
    """独立进程：只负责机器人控制"""
    robot = Robot(robot_ip)
    
    while running.value:
        try:
            target_joints = command_queue.get(timeout=0.001)
            motion = JointWaypointMotion([JointWaypoint(target_joints)], ...)
            robot.move(motion, asynchronous=True)
        except:
            pass

def main_process():
    """主进程：高频循环，不会被抢占"""
    command_queue = Queue()
    running = Value('b', True)
    
    # 启动控制进程
    proc = Process(target=control_process, args=(robot_ip, command_queue, running))
    proc.start()
    
    # 主循环 - 稳定 30Hz
    while True:
        target = compute_ik(space_mouse.control)
        command_queue.put_nowait(target)  # 非阻塞发送
        time.sleep(0.033)  # 这个 sleep 不会被抢占！
```

---

## 4. 额外发现：Discontinuity 减少

### 4.1 观察

使用多进程架构后，机器人运动的 **discontinuity（不连续/跳变）减少了**。

### 4.2 原因分析

**单进程架构的问题：**

```
时间 →
命令发送: ──A────────────────────B────────────────────C──
           ↑                    ↑                    ↑
           t=0                  t=100ms              t=200ms
                                (应该是 t=50ms!)

问题：命令发送时间不均匀，间隔从 50ms 变成 100ms
结果：机器人收到的目标位置跳变大，导致 discontinuity
```

**多进程架构的改进：**

```
时间 →
命令发送: ──A────B────C────D────E────F────G────H────
           ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
           t=0  33   66   100  133  166  200  233ms

改进：命令均匀发送，每次目标变化小
结果：机器人运动更平滑
```

### 4.3 数学解释

假设 SpaceMouse 以恒定速度推动：

| 架构 | 命令间隔 | 每次位移增量 | 运动特征 |
|------|---------|-------------|---------|
| 单进程 | 50-150ms (不稳定) | 变化大 | 跳变 |
| 多进程 | 33ms (稳定) | 恒定小增量 | 平滑 |

---

## 5. 参数解释

### 5.1 `return_when_finished` 参数

```python
JointWaypointMotion(..., return_when_finished=False)
```

| 值 | 行为 | 用途 |
|---|---|---|
| `True` | 运动完成后 `move()` 返回 | 单次运动 |
| `False` | 运动完成后不返回，等待新目标 | **连续遥操作必须用这个** |

**为什么用 `False`？**

```
return_when_finished=True:
  move(A) → 完成 → 返回 → move(B) → 完成 → 返回 → ...
                   ↑ 每次都要重新建立控制会话，开销大

return_when_finished=False:
  move(A) → 运行中 → move(B) → 运行中 → move(C) → ...
                     ↑ 无缝切换目标，开销小
```

### 5.2 `RelativeDynamicsFactor` 参数

```python
RelativeDynamicsFactor(velocity=0.25, acceleration=0.2, jerk=0.15)
```

| 参数 | 含义 | 影响 |
|---|---|---|
| velocity | 最大速度的比例 (0-1) | 越大越快 |
| acceleration | 最大加速度的比例 | 越大响应越快 |
| jerk | 最大加加速度的比例 | 越大启停越突然 |

---

## 6. 仍存在的问题

### 6.1 高频控制下的卡顿

**现象**：即使达到 20Hz 循环频率，机械臂运动仍然一卡一卡。

**原因**：franky 的 `move()` API 每次调用都会触发 Ruckig 重新规划轨迹。

```
每次 robot.move(new_target):
  1. 中断当前运动
  2. 读取当前状态 (位置, 速度, 加速度)
  3. 规划新轨迹到 new_target
  4. 开始执行新轨迹

在 20Hz 下，每 50ms 重新规划一次
→ 机器人刚开始加速就被打断
→ 表现为"卡顿"
```

**为什么 `test_multiprocess_control.py` 不卡？**

```python
# 目标是平滑正弦波
target[0] += 0.02 * sin(step * 0.1)
```

每次目标变化方向一致，机器人可以持续加速。

**为什么 SpaceMouse 遥操作卡？**

人的输入有抖动，目标方向频繁变化，机器人反复加速减速。

### 6.2 可能的解决方案

| 方案 | 描述 | 难度 |
|------|------|------|
| 降低频率 | 用 10-15Hz，给机器人更多时间 | ⭐ 简单 |
| 速度控制 | 用 `JointVelocityMotion` | ⭐⭐ 中等 |
| 位置滤波 | 平滑目标位置，减少跳变 | ⭐⭐ 中等 |
| 底层 API | 用 libfranka 的 1kHz 控制接口 | ⭐⭐⭐ 复杂 |

---

## 7. 总结

### 7.1 关键结论

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 单进程无法达到高频率 | franky 实时线程抢占 Python | ✅ 多进程架构 |
| Discontinuity 多 | 命令发送间隔不均匀 | ✅ 多进程架构（均匀发送） |
| 高频下机械臂卡顿 | `move()` 每次重新规划 | ⚠️ 需要降频或改用速度控制 |

### 7.2 推荐配置

```python
# 推荐的稳定配置
control_frequency = 10.0  # 或 15.0
dynamics = RelativeDynamicsFactor(
    velocity=0.25,
    acceleration=0.2,
    jerk=0.15
)
return_when_finished = False
```

### 7.3 文件清单

| 文件 | 用途 |
|------|------|
| `data_collection_highfreq.py` | 多进程高频数据采集 |
| `test_multiprocess_control.py` | 多进程架构验证脚本 |
| `data_collection_with_ik.py` | 原始单进程实现（含诊断代码） |

---

## 附录：进程调度原理

### Linux 调度策略

```
SCHED_FIFO (实时)
  - 先进先出调度
  - 优先级 1-99
  - 会抢占所有普通进程
  - franky 使用这个

SCHED_OTHER (普通)
  - CFS (完全公平调度器)
  - 优先级 -20 到 19 (nice值)
  - Python 默认使用这个
```

### 为什么 franky 用实时调度？

Franka 机器人需要 **1kHz 的稳定控制循环**，任何延迟都可能导致：
- 控制不稳定
- 通信超时
- 机器人进入错误状态

因此 franky 的控制线程必须有最高优先级，不能被其他任务打断。

### 多进程的隔离性

```
进程 A (主进程)          进程 B (控制进程)
├─ 独立的地址空间        ├─ 独立的地址空间
├─ 独立的线程            ├─ 独立的线程
└─ 独立的调度            └─ 包含 franky 实时线程
                              ↓
                         只影响进程 B 的普通线程
                         不影响进程 A！
```

这就是为什么多进程架构能解决频率问题的根本原因。
