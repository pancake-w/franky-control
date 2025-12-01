from argparse import ArgumentParser

from franky import (
    JointWaypointMotion,
    JointWaypoint,
    Robot,
)

from franky_control.robot.constants import FC

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default=FC.ROBOT_IP, help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    robot.relative_dynamics_factor = 0.1

    joint_motion = JointWaypointMotion(
        [
            JointWaypoint(FC.RESET_JOINTS),
        ]
    )
    robot.move(joint_motion)