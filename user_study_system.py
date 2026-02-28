import tkinter as tk
from tkinter import messagebox
import threading
import time
import numpy as np
import json
import os
from datetime import datetime
from ps_controller import PSController
from robot_controller import RobotController
from core import LegibleSharedAutonomy
from config import (UPDATE_RATE, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                   CONTROL_SPEED, SERVO_GAIN, SERVO_AHEADTIME, HOME_POSITION, GOAL_RADIUS, TASK_WEIGHT)


class UserStudyUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("User Study System - Legible Shared Autonomy")
        self.root.geometry("800x750")
        
        # Study parameters
        self.total_rounds = 10
        self.task_weights = [0, TASK_WEIGHT]  # Two conditions: 0 and config value
        self.goals = np.array([
            [650, 0, 350],
            [700, -50, 350]
        ])
        
        # Study state
        self.participant_info = {}
        self.current_round = 0
        self.round_data = []
        self.study_started = False
        self.round_in_progress = False
        
        # Robot control
        self.controller = None
        self.robot = None
        self.shared_autonomy = None
        self.running = False
        self.enabled = False
        self.connected = False
        self.target_position = np.array([500.0, -25.0, 460.0])
        
        # Data recording
        self.current_round_frames = []
        self.round_start_time = None
        self.first_input_time = None  # Track when first input occurs
        self.has_user_input = False   # Flag to track if user has started giving input
        
        self.setup_ui()
        
    def setup_ui(self):
        # ========== Participant Information Frame ==========
        frame_participant = tk.LabelFrame(self.root, text="Participant Information", 
                                         padx=15, pady=15, font=("Arial", 11, "bold"))
        frame_participant.pack(padx=10, pady=10, fill=tk.X)
        
        # Name
        tk.Label(frame_participant, text="Name:", font=("Arial", 10)).grid(row=0, column=0, sticky='w', pady=5)
        self.entry_name = tk.Entry(frame_participant, font=("Arial", 10), width=25)
        self.entry_name.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        # Age
        tk.Label(frame_participant, text="Age:", font=("Arial", 10)).grid(row=1, column=0, sticky='w', pady=5)
        self.entry_age = tk.Entry(frame_participant, font=("Arial", 10), width=25)
        self.entry_age.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        # Gender
        tk.Label(frame_participant, text="Gender:", font=("Arial", 10)).grid(row=2, column=0, sticky='w', pady=5)
        self.entry_gender = tk.Entry(frame_participant, font=("Arial", 10), width=25)
        self.entry_gender.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        
        # Participant ID (auto-generated)
        tk.Label(frame_participant, text="Participant ID:", font=("Arial", 10)).grid(row=3, column=0, sticky='w', pady=5)
        self.label_participant_id = tk.Label(frame_participant, text="Will be generated", 
                                            font=("Arial", 10), fg="blue")
        self.label_participant_id.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        
        # ========== Connection Frame ==========
        frame_conn = tk.LabelFrame(self.root, text="Robot Connection", padx=10, pady=10)
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
        
        # ========== Study Control Frame ==========
        frame_study = tk.LabelFrame(self.root, text="Study Control", padx=15, pady=15,
                                    font=("Arial", 11, "bold"))
        frame_study.pack(padx=10, pady=10, fill=tk.X)
        
        self.btn_start_study = tk.Button(frame_study, text="Start Study", 
                                         command=self.start_study, width=20, height=2,
                                         bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                         state=tk.DISABLED)
        self.btn_start_study.pack(pady=10)
        
        # Progress display
        self.label_progress = tk.Label(frame_study, 
                                       text=f"Round: 0 / {self.total_rounds}", 
                                       font=("Arial", 14, "bold"))
        self.label_progress.pack(pady=5)
        
        self.label_current_condition = tk.Label(frame_study, 
                                                text="Condition: Not started", 
                                                font=("Arial", 11))
        self.label_current_condition.pack(pady=5)
        
        # ========== Current Target Frame ==========
        frame_target = tk.LabelFrame(self.root, text="Current Target", padx=15, pady=15,
                                     font=("Arial", 11, "bold"))
        frame_target.pack(padx=10, pady=10, fill=tk.X)
        
        self.label_current_target = tk.Label(frame_target, 
                                             text="No active target", 
                                             font=("Arial", 16, "bold"),
                                             fg="#FF5722",
                                             bg="#FFF3E0",
                                             relief=tk.RAISED,
                                             padx=20,
                                             pady=15)
        self.label_current_target.pack(pady=10, fill=tk.X)
        
        # ========== Position Display Frame ==========
        frame_pos = tk.LabelFrame(self.root, text="Robot Position", padx=10, pady=10)
        frame_pos.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(frame_pos, text="", font=("Arial", 10, "bold")).grid(row=0, column=0, pady=5)
        tk.Label(frame_pos, text="Actual", font=("Arial", 10, "bold")).grid(row=0, column=1, pady=5, padx=10)
        tk.Label(frame_pos, text="Target", font=("Arial", 10, "bold")).grid(row=0, column=2, pady=5, padx=10)
        
        self.actual_labels = {}
        self.target_labels = {}
        
        for i, name in enumerate(['X', 'Y', 'Z']):
            row = i + 1
            tk.Label(frame_pos, text=f"{name}:", font=("Arial", 11, "bold")).grid(row=row, column=0, pady=5)
            
            self.actual_labels[name] = tk.Label(frame_pos, text="0.0 mm", font=("Arial", 10), fg="green")
            self.actual_labels[name].grid(row=row, column=1, pady=5, padx=10)
            
            self.target_labels[name] = tk.Label(frame_pos, text="0.0 mm", font=("Arial", 10), fg="blue")
            self.target_labels[name].grid(row=row, column=2, pady=5, padx=10)
        
        # ========== Goal Beliefs Frame ==========
        frame_belief = tk.LabelFrame(self.root, text="Goal Beliefs", padx=10, pady=10)
        frame_belief.pack(padx=10, pady=10, fill=tk.X)
        
        self.belief_labels = []
        for i in range(len(self.goals)):
            label = tk.Label(frame_belief, text=f"Goal {i+1}: 0.00", font=("Arial", 10))
            label.pack()
            self.belief_labels.append(label)
        
        # ========== Status Frame ==========
        frame_status = tk.LabelFrame(self.root, text="Status", padx=10, pady=10)
        frame_status.pack(padx=10, pady=10, fill=tk.X)
        
        self.label_status = tk.Label(frame_status, text="Ready - Please enter participant information", 
                                     font=("Arial", 10))
        self.label_status.pack()
    
    def toggle_connect(self):
        if not self.connected:
            try:
                # Validate participant info first
                name = self.entry_name.get().strip()
                age = self.entry_age.get().strip()
                gender = self.entry_gender.get().strip()
                
                if not name or not age or not gender:
                    messagebox.showerror("Error", "Please fill in all participant information")
                    return
                
                try:
                    age_int = int(age)
                    if age_int < 18 or age_int > 100:
                        messagebox.showerror("Error", "Age must be between 18 and 100")
                        return
                except ValueError:
                    messagebox.showerror("Error", "Age must be a number")
                    return
                
                # Initialize controller
                try:
                    self.controller = PSController()
                    print("Controller connected successfully")
                except RuntimeError as e:
                    messagebox.showerror("Error", f"Controller error: {e}\nPlease connect PS controller")
                    return
                
                # Connect to robot
                ip_address = self.entry_ip.get().strip()
                if not ip_address:
                    self.label_status.config(text="Error: Enter IP address")
                    return
                
                self.robot = RobotController()
                self.robot.ip = ip_address
                success = self.robot.connect()
                
                if success:
                    self.connected = True
                    current_pose = self.robot.get_pose()
                    if current_pose is None:
                        current_pose = np.array(HOME_POSITION)
                    self.controller.position = current_pose.copy()
                    self.target_position = current_pose.copy()
                    
                    # Generate participant ID
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.participant_info = {
                        'name': name,
                        'age': age_int,
                        'gender': gender,
                        'participant_id': f"P{timestamp}",
                        'start_time': datetime.now().isoformat()
                    }
                    self.label_participant_id.config(text=self.participant_info['participant_id'])
                    
                    # Disable participant info editing
                    self.entry_name.config(state=tk.DISABLED)
                    self.entry_age.config(state=tk.DISABLED)
                    self.entry_gender.config(state=tk.DISABLED)
                    
                    self.btn_connect.config(text="Disconnect")
                    self.btn_enable.config(state=tk.NORMAL)
                    self.entry_ip.config(state=tk.DISABLED)
                    
                    threading.Thread(target=self.display_update_loop, daemon=True).start()
                    
                    self.label_status.config(text=f"Connected to {ip_address} - Ready to start")
                else:
                    self.label_status.config(text=f"Failed to connect to {ip_address}")
                    
            except Exception as e:
                self.label_status.config(text=f"Connection error: {e}")
        else:
            # Disconnect
            self.running = False
            self.enabled = False
            time.sleep(0.2)
            
            if self.robot:
                self.robot.disconnect()
            if self.controller:
                self.controller.close()
            self.connected = False
            
            self.btn_connect.config(text="Connect")
            self.btn_enable.config(text="Enable", state=tk.DISABLED)
            self.entry_ip.config(state=tk.NORMAL)
            self.label_status.config(text="Disconnected")
    
    def toggle_enable(self):
        if not self.enabled:
            self.robot.enable()
            self.enabled = True
            self.btn_enable.config(text="Disable")
            self.btn_start_study.config(state=tk.NORMAL)
            self.label_status.config(text="Enabled - Click 'Start Study' to begin")
        else:
            self.running = False
            self.enabled = False
            time.sleep(0.1)
            self.robot.disable()
            self.btn_enable.config(text="Enable")
            self.btn_start_study.config(state=tk.DISABLED)
            self.label_status.config(text="Disabled")
    
    def clear_error(self):
        if self.connected and self.robot:
            self.robot.clear_error()
            self.label_status.config(text="Error cleared")
    
    def start_study(self):
        if self.study_started:
            messagebox.showinfo("Info", "Study already in progress")
            return
        
        self.study_started = True
        self.btn_start_study.config(state=tk.DISABLED)
        self.label_status.config(text="Study started - Preparing...")
        
        # Generate random sequence for all rounds
        self.generate_trial_sequence()
        
        # Start first round
        threading.Thread(target=self.run_study, daemon=True).start()
    
    def generate_trial_sequence(self):
        """Generate randomized trial sequence for 10 rounds"""
        self.trial_sequence = []
        
        # Ensure balanced design: each task_weight appears roughly equally
        task_weights_pool = self.task_weights * (self.total_rounds // len(self.task_weights))
        # Add remaining trials
        remaining = self.total_rounds - len(task_weights_pool)
        task_weights_pool.extend(self.task_weights[:remaining])
        
        # Shuffle
        np.random.shuffle(task_weights_pool)
        
        # Assign random goals
        for i, tw in enumerate(task_weights_pool):
            target_goal = np.random.randint(0, len(self.goals))
            self.trial_sequence.append({
                'round_num': i + 1,
                'task_weight': tw,
                'target_goal': target_goal
            })
        
        print(f"Generated trial sequence: {self.trial_sequence}")
    
    def run_study(self):
        """Main study loop - runs all 10 rounds"""
        for round_num in range(1, self.total_rounds + 1):
            self.current_round = round_num
            trial_config = self.trial_sequence[round_num - 1]
            
            self.update_progress_display(round_num, trial_config['task_weight'])
            
            # Move to home position
            self.move_to_home()
            time.sleep(1)
            
            # Run one round
            round_data = self.run_single_round(trial_config)
            self.round_data.append(round_data)
            
            # Wait 3 seconds at goal
            self.label_status.config(text=f"Round {round_num} complete - Please fill questionnaire")
            time.sleep(3)
            
            # Brief pause before next round
            if round_num < self.total_rounds:
                self.label_status.config(text=f"Preparing round {round_num + 1}...")
                time.sleep(2)
        
        # Study complete
        self.study_complete()
    
    def run_single_round(self, trial_config):
        """Run a single trial round"""
        round_num = trial_config['round_num']
        task_weight = trial_config['task_weight']
        target_goal_idx = trial_config['target_goal']
        
        # Update target display
        goal_position = self.goals[target_goal_idx]
        target_text = f"Goal {target_goal_idx + 1}: [{goal_position[0]:.0f}, {goal_position[1]:.0f}, {goal_position[2]:.0f}]"
        self.label_current_target.config(text=target_text)
        
        # Initialize shared autonomy for this round
        self.shared_autonomy = LegibleSharedAutonomy(self.goals, task_weight=task_weight)
        
        # Reset data recording
        self.current_round_frames = []
        self.round_start_time = datetime.now()
        self.first_input_time = None  # Reset first input time
        self.has_user_input = False   # Reset flag
        self.round_in_progress = True
        
        # Start control loop
        self.running = True
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()
        
        self.label_status.config(text=f"Round {round_num} in progress - Move to goal {target_goal_idx + 1}")
        
        # Wait until goal is reached
        target_goal = self.goals[target_goal_idx]
        while self.round_in_progress:
            if self.controller.position is not None:
                dist = np.linalg.norm(self.controller.position - target_goal)
                if dist < GOAL_RADIUS:
                    self.round_in_progress = False
            time.sleep(0.1)
        
        # Stop control
        self.running = False
        time.sleep(0.1)
        
        round_end_time = datetime.now()
        # Calculate duration from first user input
        if self.first_input_time is not None:
            duration = (round_end_time - self.first_input_time).total_seconds()
        else:
            duration = (round_end_time - self.round_start_time).total_seconds()
        
        # Compute metrics
        user_inputs = np.array([f['user_input'] for f in self.current_round_frames])
        avg_user_effort = float(np.mean(np.linalg.norm(user_inputs, axis=1)))
        
        round_data = {
            'round_num': round_num,
            'task_weight': task_weight,
            'target_goal': target_goal_idx,
            'duration': duration,
            'avg_user_effort': avg_user_effort,
            'start_time': self.round_start_time.isoformat(),
            'end_time': round_end_time.isoformat(),
            'frames': self.current_round_frames,
            'num_frames': len(self.current_round_frames)
        }
        
        print(f"Round {round_num} complete: TW={task_weight}, Goal={target_goal_idx}, Duration={duration:.2f}s")
        
        return round_data
    
    def control_loop(self):
        """Main control loop for shared autonomy"""
        while self.running and self.round_in_progress:
            try:
                if self.controller.position is None:
                    self.controller.position = np.array(HOME_POSITION)
                
                raw_input = self.controller.get_input()
                velocity = -raw_input * CONTROL_SPEED
                
                # Wait for user to start moving
                if not self.shared_autonomy.started and np.linalg.norm(velocity) < 0.01:
                    time.sleep(UPDATE_RATE)
                    continue
                
                # Track first user input for duration calculation
                if not self.has_user_input and np.linalg.norm(velocity) > 0.01:
                    self.has_user_input = True
                    self.first_input_time = datetime.now()
                    print(f"First user input detected at t={(self.first_input_time - self.round_start_time).total_seconds():.2f}s")
                
                state = self.controller.position.copy()
                user_input = velocity
                
                # Update belief and compute robot action
                self.shared_autonomy.update_belief(state, user_input)
                robot_action = self.shared_autonomy.compute_robot_action(state, user_input)
                
                # Blend actions
                if np.linalg.norm(user_input) < 0.01:
                    combined = robot_action
                else:
                    combined = self.shared_autonomy.beta * user_input + (1 - self.shared_autonomy.beta) * robot_action
                
                # Update position
                self.controller.position[0] = np.clip(
                    self.controller.position[0] + combined[0] * UPDATE_RATE, X_MIN, X_MAX)
                self.controller.position[1] = np.clip(
                    self.controller.position[1] + combined[1] * UPDATE_RATE, Y_MIN, Y_MAX)
                self.controller.position[2] = np.clip(
                    self.controller.position[2] + combined[2] * UPDATE_RATE, Z_MIN, Z_MAX)
                
                self.target_position = self.controller.position.copy()
                
                # Record frame data
                frame_data = {
                    'time': datetime.now().isoformat(),
                    'position': state.tolist(),
                    'user_input': user_input.tolist(),
                    'robot_action': robot_action.tolist(),
                    'combined_action': combined.tolist(),
                    'beliefs': self.shared_autonomy.beliefs.tolist(),
                    'beta': float(self.shared_autonomy.beta)
                }
                self.current_round_frames.append(frame_data)
                
                # Update beliefs display
                self.update_beliefs()
                
                # Send to robot
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
    
    def move_to_home(self):
        """Move robot to home position"""
        self.label_status.config(text="Moving to home position...")
        self.root.update()
        
        current_pos = self.robot.get_pose()
        if current_pos is None:
            current_pos = self.controller.position.copy()
        
        home = np.array(HOME_POSITION)
        duration = 3.0
        steps = int(duration / UPDATE_RATE)
        
        for i in range(steps):
            alpha = (i + 1) / steps
            target = current_pos * (1 - alpha) + home * alpha
            
            self.robot.servo_move(
                target[0], target[1], target[2],
                t=UPDATE_RATE,
                gain=SERVO_GAIN,
                aheadtime=SERVO_AHEADTIME
            )
            time.sleep(UPDATE_RATE)
        
        self.controller.position = np.array(HOME_POSITION)
        self.target_position = np.array(HOME_POSITION)
        
        if self.shared_autonomy:
            self.shared_autonomy.reset()
    
    def update_progress_display(self, round_num, task_weight):
        """Update progress labels"""
        self.label_progress.config(text=f"Round: {round_num} / {self.total_rounds}")
        
        if task_weight == 0:
            condition = "Standard SA (TW=0)"
        else:
            condition = f"High Legibility (TW={task_weight})"
        
        self.label_current_condition.config(text=f"Condition: {condition}")
    
    def display_update_loop(self):
        """Update position display"""
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
        """Update position labels"""
        for i, name in enumerate(['X', 'Y', 'Z']):
            actual = actual_pos[i]
            target = target_pos[i]
            
            self.actual_labels[name].config(text=f"{actual:.1f} mm")
            self.target_labels[name].config(text=f"{target:.1f} mm")
    
    def update_beliefs(self):
        """Update belief labels"""
        if self.shared_autonomy:
            for i, label in enumerate(self.belief_labels):
                label.config(text=f"Goal {i+1}: {self.shared_autonomy.beliefs[i]:.2f}")
    
    def study_complete(self):
        self.label_status.config(text="Study Complete! Saving data...")
        self.label_current_target.config(text="Study Complete!")  # Update target display
        
        output_data = {
            'participant_info': self.participant_info,
            'trial_sequence': self.trial_sequence,
            'rounds': self.round_data,
            'completion_time': datetime.now().isoformat()
        }
        
        os.makedirs('user_study_data', exist_ok=True)
        
        json_file = f"user_study_data/{self.participant_info['participant_id']}.json"
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        excel_file = self.create_excel_template()
        
        messagebox.showinfo("Study Complete", 
                          f"Data saved!\n\nJSON: {json_file}\nExcel: {excel_file}\n\n"
                          "Please fill in the scores from paper questionnaire to Excel.")
        
        self.label_status.config(text=f"Complete - Fill scores in: {excel_file}")
        self.btn_start_study.config(state=tk.DISABLED)
    
    def create_excel_template(self):
        import pandas as pd
        
        data = []
        for round_data in self.round_data:
            data.append({
                'Participant_ID': self.participant_info['participant_id'],
                'Round': round_data['round_num'],
                'Task_Weight': round_data['task_weight'],
                'Target_Goal': round_data['target_goal'],
                'Duration_sec': round_data['duration'],
                'Intuitiveness_Score': '',
                'Collaboration_Score': ''
            })
        
        df = pd.DataFrame(data)
        excel_file = f"user_study_data/Scores_{self.participant_info['participant_id']}.xlsx"
        df.to_excel(excel_file, index=False, sheet_name='Data')
        
        return excel_file
    
    def run(self):
        """Start the UI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.study_started and self.current_round > 0:
            if not messagebox.askokcancel("Quit", "Study in progress. Really quit?"):
                return
        
        self.running = False
        time.sleep(0.2)
        
        if self.connected and self.robot:
            self.robot.disconnect()
        if self.controller:
            self.controller.close()
        
        self.root.destroy()


def main():
    print("=== User Study System ===")
    print("Initializing...")
    
    ui = UserStudyUI()
    ui.run()


if __name__ == "__main__":
    main()