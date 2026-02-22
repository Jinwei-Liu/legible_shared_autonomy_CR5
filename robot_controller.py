from dobot_api import DobotApiDashboard
import numpy as np
from config import X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX


class RobotController:
    def __init__(self, ip="192.168.201.1", dash_port=29999):
        self.ip = ip
        self.dash_port = dash_port
        self.client = None
        self.connected = False
        
    def connect(self):
        if not self.client:
            self.client = DobotApiDashboard(self.ip, self.dash_port)
        
        if not self.client.socket_dobot:
            self.client = DobotApiDashboard(self.ip, self.dash_port)
            if not self.client.socket_dobot:
                raise ConnectionError(f"Failed to connect to robot at {self.ip}:{self.dash_port}")
        
        self.connected = True
        return True
        
    def disconnect(self):
        if self.connected:
            self.client.close()
            self.connected = False
    
    def enable(self):
        self.client.EnableRobot()
        return True
        
    def disable(self):
        self.client.DisableRobot()
        
    def clear_error(self):
        self.client.ClearError()
        
    def move_to(self, x, y, z, rx=-180, ry=0, rz=-90):
        x = np.clip(x, X_MIN, X_MAX)
        y = np.clip(y, Y_MIN, Y_MAX)
        z = np.clip(z, Z_MIN, Z_MAX)
        self.client.MovL_NoWait(x, y, z, rx, ry, rz, 0)
    
    def servo_move(self, x, y, z, rx=-180, ry=0, rz=-90, t=0.05, gain=500, aheadtime=50):
        x = np.clip(x, X_MIN, X_MAX)
        y = np.clip(y, Y_MIN, Y_MAX)
        z = np.clip(z, Z_MIN, Z_MAX)
        string = f"ServoP({x},{y},{z},{rx},{ry},{rz},t={t},gain={gain},aheadtime={aheadtime})"
        self.client.sendOnly(string)
        
    def get_pose(self):
        try:
            pose_str = self.client.GetPose()
            if not pose_str or "{" not in pose_str or "}" not in pose_str:
                return None
            pose = [float(x) for x in pose_str.split("{")[1].split("}")[0].split(",")]
            if len(pose) < 3:
                return None
            return np.array(pose[:3])
        except:
            return None
            
    def set_speed_ratio(self, ratio):
        self.client.SpeedFactor(ratio)
