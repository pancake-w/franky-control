
### joint_impedance_control in libfranka and franky

**Franky**

```python
from franky import Robot

robot = Robot("172.16.0.2")

robot.set_joint_impedance([600, 600, 600, 600, 250, 150, 50])
robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])

motion = LinearMotion(target_pose)
robot.move(motion)
```

**`joint_impedance_control` implemention in libfranka, only have the `setJointImpedance` and `setCartesianImpedance` methods with only Kp instead of Kd**
```cpp
// franky inherits from franka::Robot; these methods come from the official libfranka API (defined in the franka::Robot class)
void setJointImpedance(const std::array<double, 7>& K_theta);
void setCartesianImpedance(const std::array<double, 6>& K_x);
```

**Cartesian Calculation Equation：**
```
D_x = 2.0 * sqrt(K_x)
```

```cpp
// libfranka/examples/joint_impedance_control.cpp 
// Compliance parameters
const double translational_stiffness{150.0};
const double rotational_stiffness{10.0};
Eigen::MatrixXd stiffness(6, 6);
Eigen::MatrixXd damping(6, 6);
stiffness.setZero();
stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
damping.setZero();
// damping（D = 2 * sqrt(K)）
damping.topLeftCorner(3, 3) << 2.0 * sqrt(translational_stiffness) *
                                   Eigen::MatrixXd::Identity(3, 3);
damping.bottomRightCorner(3, 3) << 2.0 * sqrt(rotational_stiffness) *
                                       Eigen::MatrixXd::Identity(3, 3);
```

### Impedance Motion Control in libfranka and franky

```python
from franky import Robot, ImpedanceMotion, Affine

robot = Robot("172.16.0.2")

target_pose = Affine([0.5, 0.0, 0.3])
motion = ImpedanceMotion(target_pose)  # impedance control, using torque to control internal
motion.translational_stiffness = 2000
motion.rotational_stiffness = 200

robot.move(motion)  # will call robot.control(torque_callback) in libfranka
```

**`ImpedanceMotion` implemention in franky, but not in libfranka**
```cpp
// source: franky/src/motion/impedance_motion.cpp
ImpedanceMotion::ImpedanceMotion(Affine target, const ImpedanceMotion::Params &params) {
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << params.translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << params.rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  
  // damping（same as D = 2 * sqrt(K)）
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(params.translational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(params.rotational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
}

// calculate torques in the control loop
franka::Torques ImpedanceMotion::nextCommandImpl(...) {
  auto wrench = -stiffness * error - damping * velocity;  // impedance control law
  auto tau_d = jacobian.transpose() * wrench + coriolis;  // calculate joint torques
  return franka::Torques(tau_d_array);
}
```
