import numpy as np
import tkinter as tk
import threading
import time
import json
from datetime import datetime

from ps_controller import PSController
from robot_controller import RobotController
from core import LegibleSharedAutonomy
from config import (UPDATE_RATE, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                    CONTROL_SPEED, TASK_WEIGHT, SERVO_GAIN, SERVO_AHEADTIME,
                    HOME_POSITION)


class RecordingUI:
    def __init__(self, controller, robot, shared_autonomy, goals):
        self.controller     = controller
        self.robot          = robot
        self.shared_autonomy = shared_autonomy
        self.goals          = goals

        self.root = tk.Tk()
        self.root.title("Shared Autonomy — Record")
        self.root.geometry("720x640")

        self.running         = False
        self.enabled         = False
        self.connected       = False
        self.autonomy_mode   = tk.StringVar(value="Standard")
        self.target_position = np.array(HOME_POSITION, dtype=float)

        # ---- recording state ----
        self.recording        = False
        self.recorded_frames  = []
        self.record_start_time = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Connection
        f_conn = tk.LabelFrame(self.root, text="Connection", padx=8, pady=6)
        f_conn.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(f_conn, text="IP:").grid(row=0, column=0, padx=4)
        self.entry_ip = tk.Entry(f_conn, width=15)
        self.entry_ip.insert(0, "8.209.98.146")
        self.entry_ip.grid(row=0, column=1, padx=4)

        self.btn_connect = tk.Button(f_conn, text="Connect",
                                     command=self.toggle_connect, width=10)
        self.btn_connect.grid(row=0, column=2, padx=4)

        self.btn_enable = tk.Button(f_conn, text="Enable",
                                    command=self.toggle_enable,
                                    state=tk.DISABLED, width=10)
        self.btn_enable.grid(row=0, column=3, padx=4)

        tk.Button(f_conn, text="Clear Error",
                  command=self.clear_error, width=10).grid(row=0, column=4, padx=4)

        self.btn_reset = tk.Button(f_conn, text="Reset Home",
                                   command=self.reset_home,
                                   state=tk.DISABLED, width=10)
        self.btn_reset.grid(row=0, column=5, padx=4)

        # Control mode
        f_mode = tk.LabelFrame(self.root, text="Control Mode", padx=8, pady=6)
        f_mode.pack(padx=10, pady=5, fill=tk.X)

        tk.Radiobutton(f_mode, text="Standard SA  (TW=0)",
                       variable=self.autonomy_mode,
                       value="Standard").pack(side=tk.LEFT, padx=12)
        tk.Radiobutton(f_mode, text=f"Legible SA  (TW={TASK_WEIGHT})",
                       variable=self.autonomy_mode,
                       value="Legible").pack(side=tk.LEFT, padx=12)

        # Recording controls
        f_rec = tk.LabelFrame(self.root, text="Recording", padx=8, pady=6)
        f_rec.pack(padx=10, pady=5, fill=tk.X)

        self.btn_record = tk.Button(f_rec, text="▶  Start Recording",
                                    command=self.toggle_recording,
                                    width=18, bg="#90EE90", relief=tk.RAISED)
        self.btn_record.pack(side=tk.LEFT, padx=8)

        self.btn_save = tk.Button(f_rec, text="💾  Save Data",
                                  command=self.save_data,
                                  width=14, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=8)

        self.lbl_frames = tk.Label(f_rec, text="Frames: 0  |  Time: 0.0 s",
                                   font=("Arial", 10))
        self.lbl_frames.pack(side=tk.LEFT, padx=10)

        # Position display
        f_pos = tk.LabelFrame(self.root, text="Position", padx=8, pady=6)
        f_pos.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        for col, label in enumerate(["", "Actual", "Target", "Error"]):
            tk.Label(f_pos, text=label,
                     font=("Arial", 10, "bold")).grid(row=0, column=col,
                                                       padx=10, pady=4)

        self.actual_labels = {}
        self.target_labels = {}
        self.error_labels  = {}

        for i, axis in enumerate(["X", "Y", "Z"]):
            r = i + 1
            tk.Label(f_pos, text=f"{axis}:",
                     font=("Arial", 12, "bold")).grid(row=r, column=0, pady=4)

            self.actual_labels[axis] = tk.Label(f_pos, text="—",
                                                font=("Arial", 11), fg="green")
            self.actual_labels[axis].grid(row=r, column=1, padx=10)

            self.target_labels[axis] = tk.Label(f_pos, text="—",
                                                font=("Arial", 11), fg="blue")
            self.target_labels[axis].grid(row=r, column=2, padx=10)

            self.error_labels[axis]  = tk.Label(f_pos, text="—",
                                                font=("Arial", 11), fg="red")
            self.error_labels[axis].grid(row=r, column=3, padx=10)

        # Speed
        f_spd = tk.LabelFrame(self.root, text="Speed", padx=8, pady=4)
        f_spd.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(f_spd, text="Ratio:").grid(row=0, column=0, padx=4)
        self.speed_scale = tk.Scale(f_spd, from_=1, to=100,
                                    orient=tk.HORIZONTAL,
                                    command=self.on_speed_change, length=200)
        self.speed_scale.set(50)
        self.speed_scale.grid(row=0, column=1, padx=4)
        self.lbl_speed = tk.Label(f_spd, text="50%")
        self.lbl_speed.grid(row=0, column=2, padx=4)

        # Status
        f_stat = tk.LabelFrame(self.root, text="Status", padx=8, pady=4)
        f_stat.pack(padx=10, pady=5, fill=tk.X)
        self.lbl_status = tk.Label(f_stat, text="Ready", font=("Arial", 10))
        self.lbl_status.pack()

        # Beliefs
        f_bel = tk.LabelFrame(self.root, text="Goal Beliefs", padx=8, pady=4)
        f_bel.pack(padx=10, pady=5, fill=tk.X)
        self.belief_labels = []
        for i in range(len(self.goals)):
            lbl = tk.Label(f_bel, text=f"Goal {i+1}: 0.50",
                           font=("Arial", 10))
            lbl.pack()
            self.belief_labels.append(lbl)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def toggle_recording(self):
        if not self.recording:
            self.recording         = True
            self.recorded_frames   = []
            self.record_start_time = time.time()
            self.btn_record.config(text="⏹  Stop Recording", bg="#FF7F7F")
            self.btn_save.config(state=tk.DISABLED)
            self.lbl_status.config(text="Recording…")
        else:
            self.recording = False
            self.btn_record.config(text="▶  Start Recording", bg="#90EE90")
            n = len(self.recorded_frames)
            dur = self.recorded_frames[-1]["time"] if n else 0.0
            self.lbl_status.config(
                text=f"Stopped. {n} frames  ({dur:.2f} s)")
            if n > 0:
                self.btn_save.config(state=tk.NORMAL)

    def save_data(self):
        if not self.recorded_frames:
            self.lbl_status.config(text="No data to save!")
            return

        mode      = self.autonomy_mode.get()
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"trajectory_{mode}_{ts}.json"

        payload = {
            "autonomy_mode" : mode,
            "task_weight"   : TASK_WEIGHT if mode == "Legible" else 0,
            "goals"         : self.goals.tolist(),
            "home_position" : list(HOME_POSITION),
            "recorded_at"   : ts,
            "frames"        : self.recorded_frames,
        }

        with open(filename, "w") as fh:
            json.dump(payload, fh, indent=2)

        self.lbl_status.config(text=f"Saved → {filename}")
        self.btn_save.config(state=tk.DISABLED)
        print(f"[Saved] {filename}  ({len(self.recorded_frames)} frames)")

    # ------------------------------------------------------------------
    # Connection / enable helpers
    # ------------------------------------------------------------------
    def on_speed_change(self, value):
        self.lbl_speed.config(text=f"{value}%")
        if self.connected:
            self.robot.set_speed_ratio(int(value))

    def toggle_connect(self):
        if not self.connected:
            ip = self.entry_ip.get().strip()
            if not ip:
                self.lbl_status.config(text="Error: enter IP address")
                return
            try:
                self.robot.ip = ip
                if self.robot.connect():
                    self.connected = True
                    pose = self.robot.get_pose()
                    if pose is None:
                        pose = np.array(HOME_POSITION, dtype=float)
                    self.controller.position = pose.copy()
                    self.target_position     = pose.copy()

                    self.btn_connect.config(text="Disconnect")
                    self.btn_enable.config(state=tk.NORMAL)
                    self.btn_reset.config(state=tk.NORMAL)
                    self.entry_ip.config(state=tk.DISABLED)

                    threading.Thread(target=self._display_loop,
                                     daemon=True).start()
                    self.lbl_status.config(text=f"Connected to {ip}")
                else:
                    self.lbl_status.config(text=f"Failed to connect to {ip}")
            except Exception as e:
                self.lbl_status.config(text=f"Error: {e}")
        else:
            self.running   = False
            self.enabled   = False
            time.sleep(0.2)
            self.robot.disconnect()
            self.connected = False
            self.btn_connect.config(text="Connect")
            self.btn_enable.config(text="Enable", state=tk.DISABLED)
            self.btn_reset.config(state=tk.DISABLED)
            self.entry_ip.config(state=tk.NORMAL)
            self.lbl_status.config(text="Disconnected")

    def toggle_enable(self):
        if not self.enabled:
            self.robot.enable()
            self.enabled = True
            self.running = True
            self.robot.set_speed_ratio(self.speed_scale.get())
            self.btn_enable.config(text="Disable")
            self.lbl_status.config(text="Enabled")
            threading.Thread(target=self._control_loop, daemon=True).start()
        else:
            self.running = False
            self.enabled = False
            time.sleep(0.1)
            self.robot.disable()
            self.btn_enable.config(text="Enable")
            self.lbl_status.config(text="Disabled")

    def clear_error(self):
        if self.connected:
            self.robot.clear_error()
            self.lbl_status.config(text="Error cleared")

    def reset_home(self):
        if not self.connected:
            return
        was_running = self.running
        if self.running:
            self.running = False
            time.sleep(0.2)

        self.lbl_status.config(text="Moving to home…")
        self.root.update()

        cur = self.robot.get_pose()
        if cur is None:
            cur = self.controller.position.copy()

        home  = np.array(HOME_POSITION, dtype=float)
        steps = int(3.0 / UPDATE_RATE)
        for i in range(steps):
            alpha  = (i + 1) / steps
            target = cur * (1 - alpha) + home * alpha
            self.robot.servo_move(target[0], target[1], target[2],
                                  t=UPDATE_RATE,
                                  gain=SERVO_GAIN,
                                  aheadtime=SERVO_AHEADTIME)
            time.sleep(UPDATE_RATE)

        self.controller.position = home.copy()
        self.target_position     = home.copy()
        self.shared_autonomy.reset()
        self.lbl_status.config(text="Home reset — waiting for input")

        if was_running:
            self.running = True
            threading.Thread(target=self._control_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Control loop (runs in background thread)
    # ------------------------------------------------------------------
    def _control_loop(self):
        while self.running:
            try:
                if self.controller.position is None:
                    self.controller.position = np.array(HOME_POSITION, dtype=float)

                mode = self.autonomy_mode.get()
                self.shared_autonomy.task_weight = (TASK_WEIGHT
                                                    if mode == "Legible" else 0)

                raw_input  = self.controller.get_input()
                user_input = -raw_input * CONTROL_SPEED

                # Wait for first non-zero input
                if (not self.shared_autonomy.started
                        and np.linalg.norm(user_input) < 0.01):
                    time.sleep(UPDATE_RATE)
                    continue

                state = self.controller.position.copy()

                self.shared_autonomy.update_belief(state, user_input)
                robot_action = self.shared_autonomy.compute_robot_action(
                    state, user_input)

                if np.linalg.norm(user_input) < 0.01:
                    combined = robot_action
                else:
                    b = self.shared_autonomy.beta
                    combined = b * user_input + (1 - b) * robot_action

                # Integrate position
                new_pos = self.controller.position.copy()
                new_pos[0] = np.clip(
                    new_pos[0] + combined[0] * UPDATE_RATE, X_MIN, X_MAX)
                new_pos[1] = np.clip(
                    new_pos[1] + combined[1] * UPDATE_RATE, Y_MIN, Y_MAX)
                new_pos[2] = np.clip(
                    new_pos[2] + combined[2] * UPDATE_RATE, Z_MIN, Z_MAX)
                self.controller.position = new_pos
                self.target_position     = new_pos.copy()

                # ---- record one frame ----
                if self.recording:
                    elapsed = time.time() - self.record_start_time
                    self.recorded_frames.append({
                        "time"        : round(elapsed, 4),
                        "position"    : new_pos.tolist(),
                        "beliefs"     : self.shared_autonomy.beliefs.tolist(),
                        "user_input"  : user_input.tolist(),
                        "robot_action": robot_action.tolist(),
                        "combined"    : combined.tolist(),
                        "beta"        : float(self.shared_autonomy.beta),
                    })
                    n = len(self.recorded_frames)
                    if n % 20 == 0:
                        dur = elapsed
                        self.root.after(
                            0,
                            lambda n=n, d=dur:
                                self.lbl_frames.config(
                                    text=f"Frames: {n}  |  Time: {d:.1f} s"))

                # Update belief labels
                self._update_belief_display()

                # Send to robot
                self.robot.servo_move(
                    self.target_position[0],
                    self.target_position[1],
                    self.target_position[2],
                    t=UPDATE_RATE,
                    gain=SERVO_GAIN,
                    aheadtime=SERVO_AHEADTIME,
                )

                time.sleep(UPDATE_RATE)

            except Exception as e:
                print(f"Control error: {e}")
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Display loop
    # ------------------------------------------------------------------
    def _display_loop(self):
        while self.connected:
            try:
                actual = self.robot.get_pose()
                if actual is not None:
                    for i, axis in enumerate(["X", "Y", "Z"]):
                        a = actual[i]
                        t = self.target_position[i]
                        self.actual_labels[axis].config(text=f"{a:.1f} mm")
                        self.target_labels[axis].config(text=f"{t:.1f} mm")
                        self.error_labels[axis].config(text=f"{t-a:.1f} mm")
                time.sleep(0.1)
            except Exception as e:
                print(f"Display error: {e}")
                time.sleep(0.2)

    def _update_belief_display(self):
        for i, lbl in enumerate(self.belief_labels):
            lbl.config(
                text=f"Goal {i+1}: {self.shared_autonomy.beliefs[i]:.2f}")

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        # Auto-save if still recording
        if self.recording and self.recorded_frames:
            self.recording = False
            self.save_data()

        self.running = False
        time.sleep(0.2)
        if self.connected:
            self.robot.disconnect()
        self.controller.close()
        self.root.destroy()


# ======================================================================
# Entry point
# ======================================================================
def main():
    print("=== Shared Autonomy  (with recording) ===")

    goals = np.array([
        [687.2, 48.6, 372.8],
        [700.0,-21.8,372.8]
    ])

    try:
        controller = PSController()
        print("Controller connected.")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please connect a PS controller and try again.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    try:
        robot           = RobotController()
        shared_autonomy = LegibleSharedAutonomy(goals)

        print("\nGoals:")
        for i, g in enumerate(goals):
            print(f"  Goal {i+1}: X={g[0]}, Y={g[1]}, Z={g[2]}")

        ui = RecordingUI(controller, robot, shared_autonomy, goals)
        ui.run()

    except Exception as e:
        print(f"Error: {e}")
        controller.close()


if __name__ == "__main__":
    main()
