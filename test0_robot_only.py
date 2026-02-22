import tkinter as tk
from tkinter import messagebox
from robot_controller import RobotController
import time
import threading


class RobotTestUI:
    def __init__(self):
        self.robot = RobotController()
        self.connected = False
        self.enabled = False
        
        self.root = tk.Tk()
        self.root.title("Robot Connection Test")
        self.root.geometry("500x400")
        
        self.setup_ui()
        
    def setup_ui(self):
        frame_conn = tk.LabelFrame(self.root, text="Connection", padx=10, pady=10)
        frame_conn.pack(padx=10, pady=10, fill=tk.X)
        
        tk.Label(frame_conn, text="IP:").grid(row=0, column=0)
        self.entry_ip = tk.Entry(frame_conn, width=15)
        self.entry_ip.insert(0, "192.168.201.1")
        self.entry_ip.grid(row=0, column=1, padx=5)
        
        self.btn_connect = tk.Button(frame_conn, text="Connect", command=self.toggle_connect)
        self.btn_connect.grid(row=0, column=2, padx=5)
        
        self.btn_enable = tk.Button(frame_conn, text="Enable", command=self.toggle_enable, state=tk.DISABLED)
        self.btn_enable.grid(row=0, column=3, padx=5)
        
        tk.Button(frame_conn, text="Clear Error", command=self.clear_error).grid(row=0, column=4, padx=5)
        
        frame_pos = tk.LabelFrame(self.root, text="Current Position", padx=10, pady=10)
        frame_pos.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.labels = {}
        for i, name in enumerate(['X', 'Y', 'Z']):
            tk.Label(frame_pos, text=f"{name}:", font=("Arial", 14, "bold")).grid(row=i, column=0, sticky=tk.W, pady=10, padx=10)
            self.labels[name] = tk.Label(frame_pos, text="---", font=("Arial", 14))
            self.labels[name].grid(row=i, column=1, sticky=tk.W, pady=10)
            tk.Label(frame_pos, text="mm", font=("Arial", 14)).grid(row=i, column=2, sticky=tk.W, pady=10)
        
        frame_manual = tk.LabelFrame(self.root, text="Manual Control", padx=10, pady=10)
        frame_manual.pack(padx=10, pady=10, fill=tk.X)
        
        tk.Label(frame_manual, text="Target X:").grid(row=0, column=0)
        self.entry_x = tk.Entry(frame_manual, width=8)
        self.entry_x.insert(0, "500")
        self.entry_x.grid(row=0, column=1, padx=5)
        
        tk.Label(frame_manual, text="Y:").grid(row=0, column=2)
        self.entry_y = tk.Entry(frame_manual, width=8)
        self.entry_y.insert(0, "0")
        self.entry_y.grid(row=0, column=3, padx=5)
        
        tk.Label(frame_manual, text="Z:").grid(row=0, column=4)
        self.entry_z = tk.Entry(frame_manual, width=8)
        self.entry_z.insert(0, "460")
        self.entry_z.grid(row=0, column=5, padx=5)
        
        tk.Button(frame_manual, text="Move", command=self.manual_move).grid(row=0, column=6, padx=5)
        
        frame_status = tk.LabelFrame(self.root, text="Status", padx=10, pady=10)
        frame_status.pack(padx=10, pady=10, fill=tk.X)
        
        self.label_status = tk.Label(frame_status, text="Ready", font=("Arial", 10))
        self.label_status.pack()
        
    def toggle_connect(self):
        if not self.connected:
            try:
                self.label_status.config(text="Connecting...")
                self.root.update()
                
                self.robot.connect()
                self.connected = True
                
                current_pose = self.robot.get_pose()
                
                self.btn_connect.config(text="Disconnect")
                self.btn_enable.config(state=tk.NORMAL)
                self.label_status.config(text=f"Connected - Robot at X={current_pose[0]:.1f} Y={current_pose[1]:.1f} Z={current_pose[2]:.1f}")
                
                self.labels['X'].config(text=f"{current_pose[0]:.1f}")
                self.labels['Y'].config(text=f"{current_pose[1]:.1f}")
                self.labels['Z'].config(text=f"{current_pose[2]:.1f}")
                
                self.entry_x.delete(0, tk.END)
                self.entry_x.insert(0, f"{current_pose[0]:.1f}")
                self.entry_y.delete(0, tk.END)
                self.entry_y.insert(0, f"{current_pose[1]:.1f}")
                self.entry_z.delete(0, tk.END)
                self.entry_z.insert(0, f"{current_pose[2]:.1f}")
                
            except Exception as e:
                self.label_status.config(text=f"Connection failed: {e}")
                messagebox.showerror("Connection Error", str(e))
        else:
            self.robot.disconnect()
            self.connected = False
            self.enabled = False
            self.btn_connect.config(text="Connect")
            self.btn_enable.config(text="Enable", state=tk.DISABLED)
            self.label_status.config(text="Disconnected")
    
    def toggle_enable(self):
        if not self.enabled:
            try:
                self.robot.enable()
                self.enabled = True
                self.btn_enable.config(text="Disable")
                self.label_status.config(text="Enabled")
            except Exception as e:
                self.label_status.config(text=f"Enable failed: {e}")
        else:
            self.robot.disable()
            self.enabled = False
            self.btn_enable.config(text="Enable")
            self.label_status.config(text="Disabled")
    
    def clear_error(self):
        if self.connected:
            try:
                self.robot.clear_error()
                self.label_status.config(text="Error cleared")
            except Exception as e:
                self.label_status.config(text=f"Clear error failed: {e}")
    
    def manual_move(self):
        if not self.enabled:
            messagebox.showwarning("Not Enabled", "Please enable robot first")
            return
        
        try:
            x = float(self.entry_x.get())
            y = float(self.entry_y.get())
            z = float(self.entry_z.get())
            
            self.label_status.config(text=f"Moving to X={x} Y={y} Z={z}...")
            self.robot.move_to(x, y, z)
            
            time.sleep(0.5)
            current_pose = self.robot.get_pose()
            self.labels['X'].config(text=f"{current_pose[0]:.1f}")
            self.labels['Y'].config(text=f"{current_pose[1]:.1f}")
            self.labels['Z'].config(text=f"{current_pose[2]:.1f}")
            
            self.label_status.config(text="Move completed")
        except Exception as e:
            self.label_status.config(text=f"Move failed: {e}")
            messagebox.showerror("Move Error", str(e))
    
    def run(self):
        self.root.mainloop()
        if self.connected:
            self.robot.disconnect()


def main():
    print("=== Robot Connection Test (No Controller Required) ===")
    print("This test validates robot connection and reads current position")
    
    app = RobotTestUI()
    app.run()


if __name__ == "__main__":
    main()
