from ps_controller import PSController
from robot_controller import RobotController
from ui_teleoperation import TeleoperationUI


def main():
    print("=== Robot Teleoperation ===")
    
    try:
        controller = PSController()
        print("Controller: OK")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("PS Controller required. Use test0_robot_only.py instead.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    try:
        robot = RobotController()
        print("Starting UI...")
        ui = TeleoperationUI(controller, robot)
        ui.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
