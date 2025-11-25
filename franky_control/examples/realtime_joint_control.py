import time
from franky import *


if __name__ == "__main__":
    robot = Robot("172.16.0.2")
    robot.relative_dynamics_factor = RelativeDynamicsFactor(
        velocity=0.2, acceleration=0.1, jerk=0.1
    )
    frequency = 20.0

    q_start = list(robot.current_joint_state.position)
    print(f"Initial position: {[f'{q:.3f}' for q in q_start]}\n")

    TEST_ITERATIONS = 50
    joint_index = 7

    i = 0
    while i < TEST_ITERATIONS:
        delta_joints = [0.0] * 7
        delta_joints[joint_index - 1] = 0.03

        current_q = list(robot.current_joint_state.position)
        # compute the target (set) position for this waypoint (relative)
        set_position = [c + d for c, d in zip(current_q, delta_joints)]
        waypoint = JointWaypoint(delta_joints, reference_type=ReferenceType.Relative)
        waypoint_motion = JointWaypointMotion([waypoint])

        robot.move(waypoint_motion, asynchronous=True)
        time.sleep(1.0 / frequency)
        
        # Get the actual position after the move
        actual_q = list(robot.current_joint_state.position)
        diff = [s - a for s, a in zip(set_position, actual_q)]
        
        print(f"Iteration {i+1}: Sent new waypoint.")
        print(f"  Set position: {set_position[joint_index-1]:.3f}")
        print(f"  Actual position: {actual_q[joint_index-1]:.3f}")
        print(f"  Difference: {diff[joint_index-1]:.3f}")
        
        i += 1

    robot.join_motion()

    joint_motion = JointWaypointMotion(
        [
            JointWaypoint([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]),
        ]
    )
    robot.move(joint_motion)
