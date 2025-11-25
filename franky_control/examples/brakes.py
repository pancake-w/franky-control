import argparse
from franky import RobotWebSession


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unlock the brakes of a Franka robot.")
    parser.add_argument("host", type=str, help="FCI IP of the robot")
    parser.add_argument("user", type=str, help="Login username of Franka desk")
    parser.add_argument("password", type=str, help="Login password of Franka Desk.")

    args = parser.parse_args()

    with RobotWebSession(args.host, args.user, args.password) as robot_web_session:
        # robot_web_session.lock_brakes()
        robot_web_session.unlock_brakes()
