#!/usr/bin/env python3
"""
Load and Collision Configuration Example
Demonstrates how to configure end-effector load and collision behavior.
"""

from argparse import ArgumentParser
import numpy as np
from franky import Robot, CartesianMotion, Affine, ReferenceType


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.05

    # ============================================
    # Example 1: Configure End-Effector Load
    # ============================================
    print("\n=== Configuring End-Effector Load ===")
    
    # Set gripper load parameters
    gripper_mass = 0.73  # kg
    gripper_cog = [0.0, 0.0, 0.05]  # center of gravity [x, y, z] in meters
    # Inertia tensor (3x3 matrix): [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
    gripper_inertia = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
    
    print(f"Setting gripper load: {gripper_mass} kg")
    robot.set_load(gripper_mass, gripper_cog, gripper_inertia)
    
    # Set end-effector offset (gripper length)
    ee_offset = Affine([0.0, 0.0, 0.1])  # 10cm offset in Z direction
    print(f"Setting EE offset: 10cm in Z direction")
    robot.set_ee(ee_offset.matrix.ravel(order='F').tolist())
    
    # Test movement with load configured
    print("\nTesting movement with configured load...")
    motion = CartesianMotion(Affine([0.0, 0.05, 0.0]), ReferenceType.Relative)
    robot.move(motion)
    robot.move(CartesianMotion(Affine([0.0, -0.05, 0.0]), ReferenceType.Relative))
    print("Movement completed")
    
    # Reset load to zero
    print("\nResetting load to zero...")
    robot.set_load(0.0, [0.0, 0.0, 0.0], [0.0]*9)
    identity_matrix = np.eye(4).ravel(order='F').tolist()
    robot.set_ee(identity_matrix)
    print("Load reset")
    
    # ============================================
    # Example 2: Configure Collision Behavior
    # ============================================
    print("\n=== Configuring Collision Behavior ===")
    
    # Set sensitive collision detection (low thresholds)
    print("Setting sensitive collision detection...")
    robot.set_collision_behavior(
        torque_thresholds=15.0,  # N·m
        force_thresholds=20.0     # N
    )
    print("Torque threshold: 15 N·m")
    print("Force threshold: 20 N")
    print("Robot is now more sensitive to collisions")
    
    # Set normal collision detection (default thresholds)
    print("\nSetting normal collision detection...")
    robot.set_collision_behavior(
        torque_thresholds=20.0,  # N·m
        force_thresholds=30.0     # N
    )
    print("Torque threshold: 20 N·m")
    print("Force threshold: 30 N")
    print("Normal collision sensitivity restored")
    
    # Advanced: Set different upper and lower thresholds
    print("\nSetting asymmetric collision thresholds...")
    robot.set_collision_behavior(
        lower_torque_threshold=10.0,
        upper_torque_threshold=25.0,
        lower_force_threshold=15.0,
        upper_force_threshold=35.0
    )
    print("Lower torque: 10 N·m, Upper torque: 25 N·m")
    print("Lower force: 15 N, Upper force: 35 N")
    
    # Restore default collision behavior
    print("\nRestoring default collision behavior...")
    robot.set_collision_behavior(
        torque_thresholds=20.0,
        force_thresholds=30.0
    )
    print("Default collision behavior restored")
