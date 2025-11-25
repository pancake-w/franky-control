#!/usr/bin/env python3
"""
Impedance Control Example
Demonstrates how to adjust joint and Cartesian impedance parameters.
"""

from argparse import ArgumentParser
from franky import Robot, JointMotion, CartesianMotion, Affine, ReferenceType


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.05

    # Get initial position
    initial_pos = robot.current_joint_state.position
    
    # ============================================
    # Example 1: Joint Impedance Control
    # ============================================
    print("\n=== Joint Impedance Control ===")
    
    # Set soft joint stiffness
    print("Setting soft joint stiffness...")
    soft_stiffness = [600, 600, 600, 600, 250, 150, 50]
    robot.set_joint_impedance(soft_stiffness)
    
    # Move with soft impedance
    target = list(initial_pos)
    target[0] += 0.2
    robot.move(JointMotion(target))
    print("Movement completed with soft impedance")
    
    # Set stiff joint stiffness
    print("\nSetting stiff joint stiffness...")
    stiff_stiffness = [5000, 5000, 5000, 5000, 4000, 3000, 3000]
    robot.set_joint_impedance(stiff_stiffness)
    
    # Move back with stiff impedance
    robot.move(JointMotion(initial_pos))
    print("Movement completed with stiff impedance")
    
    # ============================================
    # Example 2: Cartesian Impedance Control
    # ============================================
    print("\n=== Cartesian Impedance Control ===")
    
    # Set default Cartesian stiffness
    print("Setting default Cartesian stiffness...")
    default_stiffness = [3000, 3000, 3000, 300, 300, 300]
    robot.set_cartesian_impedance(default_stiffness)
    
    # Move forward
    motion = CartesianMotion(Affine([0.1, 0.0, 0.0]), ReferenceType.Relative)
    robot.move(motion)
    print("Moved forward with default stiffness")
    
    # Set soft Z-axis for compliance
    print("\nSetting soft Z-axis stiffness...")
    soft_z = [3000, 3000, 500, 300, 300, 300]
    robot.set_cartesian_impedance(soft_z)
    
    # Move down
    motion = CartesianMotion(Affine([0.0, 0.0, -0.05]), ReferenceType.Relative)
    robot.move(motion)
    print("Moved down with soft Z-axis")
    
    # Move back to initial position
    motion = CartesianMotion(Affine([-0.1, 0.0, 0.05]), ReferenceType.Relative)
    robot.move(motion)
    print("Returned to initial position")
    
    # ============================================
    # Restore Default Settings
    # ============================================
    print("\n=== Restoring Default Settings ===")
    robot.set_joint_impedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
    robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])
    print("Default impedance parameters restored")
