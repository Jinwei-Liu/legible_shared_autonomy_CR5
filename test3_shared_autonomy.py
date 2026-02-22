import numpy as np
from ps_controller import PSController
from robot_controller import RobotController
from ui_teleoperation import TeleoperationUI
from core import LegibleSharedAutonomy


def main():
    print("=== Shared Autonomy Test ===")
    print("Initializing...")
    
    goals = np.array([
        [400, 0],
        [600, 0]
    ])
    
    try:
        controller = PSController()
        print("Controller connected successfully")
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("\nPS Controller is required for this test.")
        print("Please connect your PS controller and try again.")
        print("\nAlternatively, use test0_robot_only.py to test without controller.")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print(f"Error type: {type(e)}")
        return
    
    try:
        robot = RobotController()
        shared_autonomy = LegibleSharedAutonomy(goals)
        
        print("\nGoals configured:")
        print(f"  Goal 1: X={goals[0][0]}, Y={goals[0][1]}")
        print(f"  Goal 2: X={goals[1][0]}, Y={goals[1][1]}")
        print("\nStarting UI...")
        print("  Manual mode: TW=0 (task-oriented)")
        print("  Assisted mode: TW=5 (legibility-aware)")
        
        ui = TeleoperationUI(controller, robot, shared_autonomy)
        ui.run()
        
    except Exception as e:
        print(f"Error: {e}")
        controller.close()


if __name__ == "__main__":
    main()
