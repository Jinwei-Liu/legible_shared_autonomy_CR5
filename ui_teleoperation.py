import tkinter as tk
import threading
import time
import numpy as np
from config import UPDATE_RATE, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, CONTROL_SPEED, TASK_WEIGHT, SERVO_GAIN, SERVO_AHEADTIME


class TeleoperationUI:
    def __init__(self, controller, robot, shared_autonomy=None):
        self.controller = controller
        self.robot = robot
        self.shared_autonomy = shared_autonomy
        
        self.root = tk.Tk()
        self.root.title("Robot Teleoperation")
        self.root.geometry("700x550")
        
        self.running = False
        self.enabled = False
        self.connected = False
        self.autonomy_mode = tk.StringVar(value="Manual")
        self.target_position = np.array([500.0, -25.0, 460.0])
        
        self.setup_ui()
        
    def setup_ui(self):
        frame_conn = tk.LabelFrame(self.root, text="Connection", padx=10, pady=10)
        frame_conn.pack(padx=10, pady=10, fill=tk.X)
        
        tk.Label(frame_conn, text="IP:").grid(row=0, column=0, padx=5)
        self.entry_ip = tk.Entry(frame_conn, width=15)
        self.entry_ip.insert(0, "8.209.98.146")
        self.entry_ip.grid(row=0, column=1, padx=5)
        
        self.btn_connect = tk.Button(frame_conn, text="Connect", command=self.toggle_connect, width=10)
        self.btn_connect.grid(row=0, column=2, padx=5)
        
        self.btn_enable = tk.Button(frame_conn, text="Enable", command=self.toggle_enable, 
                                     state=tk.DISABLED, width=10)
        self.btn_enable.grid(row=0, column=3, padx=5)
        
        tk.Button(frame_conn, text="Clear Error", command=self.clear_error, width=10).grid(
            row=0, column=4, padx=5)
        
        if self.shared_autonomy:
            frame_mode = tk.LabelFrame(self.root, text="Control Mode", padx=10, pady=10)
            frame_mode.pack(padx=10, pady=10, fill=tk.X)
            
            tk.Radiobutton(frame_mode, text="Manual (TW=0)", variable=self.autonomy_mode, 
                          value="Manual").pack(side=tk.LEFT, padx=10)
            tk.Radiobutton(frame_mode, text=f"Assisted (TW={TASK_WEIGHT})", variable=self.autonomy_mode,
                          value="Assisted").pack(side=tk.LEFT, padx=10)
        
        frame_pos = tk.LabelFrame(self.root, text="Position", padx=10, pady=10)
        frame_pos.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(frame_pos, text="", font=("Arial", 10, "bold")).grid(row=0, column=0, pady=5)
        tk.Label(frame_pos, text="Actual", font=("Arial", 10, "bold")).grid(row=0, column=1, pady=5, padx=10)
        tk.Label(frame_pos, text="Target", font=("Arial", 10, "bold")).grid(row=0, column=2, pady=5, padx=10)
        tk.Label(frame_pos, text="Error", font=("Arial", 10, "bold")).grid(row=0, column=3, pady=5, padx=10)
        
        self.actual_labels = {}
        self.target_labels = {}
        self.error_labels = {}
        
        for i, name in enumerate(['X', 'Y', 'Z']):
            row = i + 1
            tk.Label(frame_pos, text=f"{name}:", font=("Arial", 12, "bold")).grid(row=row, column=0, pady=5)
            
            self.actual_labels[name] = tk.Label(frame_pos, text="0.0 mm", font=("Arial", 11), fg="green")
            self.actual_labels[name].grid(row=row, column=1, pady=5, padx=10)
            
            self.target_labels[name] = tk.Label(frame_pos, text="0.0 mm", font=("Arial", 11), fg="blue")
            self.target_labels[name].grid(row=row, column=2, pady=5, padx=10)
            
            self.error_labels[name] = tk.Label(frame_pos, text="0.0 mm", font=("Arial", 11), fg="red")
            self.error_labels[name].grid(row=row, column=3, pady=5, padx=10)
        
        frame_speed = tk.LabelFrame(self.root, text="Speed", padx=10, pady=10)
        frame_speed.pack(padx=10, pady=10, fill=tk.X)
        
        tk.Label(frame_speed, text="Ratio:").grid(row=0, column=0, padx=5)
        self.speed_scale = tk.Scale(frame_speed, from_=1, to=100, orient=tk.HORIZONTAL,
                                   command=self.on_speed_change, length=200)
        self.speed_scale.set(50)
        self.speed_scale.grid(row=0, column=1, padx=5)
        self.speed_label = tk.Label(frame_speed, text="50%")
        self.speed_label.grid(row=0, column=2, padx=5)
        
        frame_status = tk.LabelFrame(self.root, text="Status", padx=10, pady=10)
        frame_status.pack(padx=10, pady=10, fill=tk.X)
        
        self.label_status = tk.Label(frame_status, text="Ready", font=("Arial", 10))
        self.label_status.pack()
        
        if self.shared_autonomy:
            frame_belief = tk.LabelFrame(self.root, text="Goal Beliefs", padx=10, pady=10)
            frame_belief.pack(padx=10, pady=10, fill=tk.X)
            
            self.belief_labels = []
            for i in range(len(self.shared_autonomy.goals)):
                label = tk.Label(frame_belief, text=f"Goal {i+1}: 0.00", font=("Arial", 10))
                label.pack()
                self.belief_labels.append(label)
    
    def on_speed_change(self, value):
        self.speed_label.config(text=f"{value}%")
        if self.connected:
            self.robot.set_speed_ratio(int(value))
    
    def toggle_connect(self):
        if not self.connected:
            try:
                ip_address = self.entry_ip.get().strip()
                if not ip_address:
                    self.label_status.config(text="Error: Enter IP address")
                    return
                
                self.robot.ip = ip_address
                success = self.robot.connect()
                
                if success:
                    self.connected = True
                    current_pose = self.robot.get_pose()
                    self.controller.position = current_pose.copy()
                    self.target_position = current_pose.copy()
                    
                    self.btn_connect.config(text="Disconnect")
                    self.btn_enable.config(state=tk.NORMAL)
                    self.entry_ip.config(state=tk.DISABLED)
                    
                    threading.Thread(target=self.display_update_loop, daemon=True).start()
                    
                    self.label_status.config(text=f"Connected to {ip_address}")
                else:
                    self.label_status.config(text=f"Failed to connect to {ip_address}")
                    
            except Exception as e:
                self.label_status.config(text=f"Connection error: {e}")
        else:
            self.running = False
            self.enabled = False
            time.sleep(0.2)
            
            self.robot.disconnect()
            self.connected = False
            
            self.btn_connect.config(text="Connect")
            self.btn_enable.config(text="Enable", state=tk.DISABLED)
            self.entry_ip.config(state=tk.NORMAL)
            self.label_status.config(text="Disconnected")
    
    def toggle_enable(self):
        if not self.enabled:
            self.robot.enable()
            self.enabled = True
            self.running = True
            
            speed_value = self.speed_scale.get()
            self.robot.set_speed_ratio(speed_value)
            
            self.btn_enable.config(text="Disable")
            self.label_status.config(text="Enabled")
            
            threading.Thread(target=self.control_loop, daemon=True).start()
        else:
            self.running = False
            self.enabled = False
            time.sleep(0.1)
            
            self.robot.disable()
            self.btn_enable.config(text="Enable")
            self.label_status.config(text="Disabled")
    
    def clear_error(self):
        if self.connected:
            self.robot.clear_error()
            self.label_status.config(text="Error cleared")
    
    def control_loop(self):
        while self.running:
            try:
                if self.shared_autonomy and self.autonomy_mode.get() == "Assisted":
                    self.shared_autonomy.task_weight = TASK_WEIGHT
                    
                    raw_input = self.controller.get_input()
                    velocity = -raw_input * CONTROL_SPEED
                    
                    state = self.controller.position[:2].copy()
                    user_input = velocity[:2]
                    
                    self.shared_autonomy.update_belief(state, user_input)
                    robot_action = self.shared_autonomy.compute_robot_action(state, user_input)
                    
                    combined = self.shared_autonomy.beta * user_input + (1 - self.shared_autonomy.beta) * robot_action
                    
                    self.controller.position[0] = np.clip(
                        self.controller.position[0] + combined[0] * UPDATE_RATE, X_MIN, X_MAX)
                    self.controller.position[1] = np.clip(
                        self.controller.position[1] + combined[1] * UPDATE_RATE, Y_MIN, Y_MAX)
                    self.controller.position[2] = np.clip(
                        self.controller.position[2] + velocity[2] * UPDATE_RATE, Z_MIN, Z_MAX)
                    
                    self.target_position = self.controller.position.copy()
                    self.update_beliefs()
                    
                elif self.shared_autonomy and self.autonomy_mode.get() == "Manual":
                    self.shared_autonomy.task_weight = 0
                    position, velocity = self.controller.update_position(UPDATE_RATE)
                    self.target_position = position.copy()
                    
                else:
                    position, velocity = self.controller.update_position(UPDATE_RATE)
                    self.target_position = position.copy()
                
                self.robot.servo_move(
                    self.target_position[0], 
                    self.target_position[1], 
                    self.target_position[2],
                    t=UPDATE_RATE,
                    gain=SERVO_GAIN,
                    aheadtime=SERVO_AHEADTIME
                )
                
                time.sleep(UPDATE_RATE)
                
            except Exception as e:
                print(f"Control error: {e}")
                time.sleep(0.1)
    
    def display_update_loop(self):
        while self.connected:
            try:
                actual_position = self.robot.get_pose()
                if actual_position is not None:
                    self.update_display(actual_position, self.target_position)
                time.sleep(0.1)
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(0.2)
    
    def update_display(self, actual_pos, target_pos):
        for i, name in enumerate(['X', 'Y', 'Z']):
            actual = actual_pos[i]
            target = target_pos[i]
            error = target - actual
            
            self.actual_labels[name].config(text=f"{actual:.1f} mm")
            self.target_labels[name].config(text=f"{target:.1f} mm")
            self.error_labels[name].config(text=f"{error:.1f} mm")
    
    def update_beliefs(self):
        if self.shared_autonomy:
            for i, label in enumerate(self.belief_labels):
                label.config(text=f"Goal {i+1}: {self.shared_autonomy.beliefs[i]:.2f}")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.running = False
        time.sleep(0.2)
        
        if self.connected:
            self.robot.disconnect()
        self.controller.close()
        
        self.root.destroy()
