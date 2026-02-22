import pygame
import numpy as np
from config import JOYSTICK_DEADZONE, CONTROL_SPEED, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


class PSController:
    def __init__(self, initial_position=None):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected. Please connect PS controller.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        if initial_position is not None:
            self.position = np.array(initial_position, dtype=float)
        else:
            self.position = np.array([500.0, -25.0, 460.0])
        
    def apply_deadzone(self, value):
        return value if abs(value) > JOYSTICK_DEADZONE else 0.0
    
    def get_input(self):
        pygame.event.pump()
        
        left_x = self.apply_deadzone(self.joystick.get_axis(0))
        left_y = self.apply_deadzone(self.joystick.get_axis(1))
        right_y = self.apply_deadzone(self.joystick.get_axis(3))
        
        return np.array([left_y, left_x, right_y])
    
    def update_position(self, dt):
        raw_input = self.get_input()
        velocity = -raw_input * CONTROL_SPEED
        
        self.position += velocity * dt
        self.position[0] = np.clip(self.position[0], X_MIN, X_MAX)
        self.position[1] = np.clip(self.position[1], Y_MIN, Y_MAX)
        self.position[2] = np.clip(self.position[2], Z_MIN, Z_MAX)
        
        return self.position.copy(), velocity
    
    def get_button(self, button_id):
        return self.joystick.get_button(button_id)
    
    def close(self):
        pygame.quit()
