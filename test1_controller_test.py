import time
import numpy as np
from ps_controller import PSController


def main():
    print("=== PS Controller Test ===")
    print("Left stick: X-Y control")
    print("Right stick Y: Z control")
    print("Press X button to exit\n")
    
    try:
        controller = PSController()
        print("Controller initialized successfully\n")
        
        while True:
            position, velocity = controller.update_position(0.05)
            
            print(f"\rPosition: X={position[0]:.1f} Y={position[1]:.1f} Z={position[2]:.1f} | "
                  f"Velocity: X={velocity[0]:.1f} Y={velocity[1]:.1f} Z={velocity[2]:.1f}", end="")
            
            if controller.get_button(0):
                print("\n\nExiting...")
                break
                
            time.sleep(0.05)
            
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
