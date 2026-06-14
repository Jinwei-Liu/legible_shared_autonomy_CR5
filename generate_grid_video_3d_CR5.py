#!/usr/bin/env python3
"""
Generate combined 3×5 grid 3D trajectory animation video (MP4)
Shows all 15 participants' 3D trajectories simultaneously in a 3-row × 5-col layout.
Each participant's subplot freezes at their last frame when their trajectories end.
The video ends only after all participants' trajectories are complete.

Requirements (install one of these):
    pip install imageio[ffmpeg]          <- recommended, bundles its own ffmpeg
    conda install -c conda-forge opencv  <- alternative
"""

# Set non-interactive Agg backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa – registers 3D projection
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Workspace boundaries (mm)
X_MIN, X_MAX = 300, 700
Y_MIN, Y_MAX = -200, 150
Z_MIN, Z_MAX = 320, 600

# Goal positions (from user_study_system.py)
GOAL_POSITIONS = {
    0: np.array([650,  0,  350], dtype=float),   # Goal 0
    1: np.array([700, -50, 350], dtype=float),   # Goal 1
}

HOME_POSITION = np.array([300.0, -25.0, 500.0])

# Colors
GOAL_COLORS = {0: '#3C5488', 1: '#F39B7F'}          # Blue, Salmon
TASK_WEIGHT_COLORS = {0: '#E64B35', 1: '#00A087'}   # Red, Teal
TRAJECTORY_COLOR = '#7E6148'
START_COLOR = '#00A087'

# Animation parameters
FPS = 15
TRAIL_LENGTH = 15
SPEED_MULTIPLIER = 3

# Fixed camera (consistent across all subplots in grid)
CAMERA_ELEV = 28
CAMERA_AZIM = 40

# Grid layout
GRID_ROWS = 3
GRID_COLS = 5


# ============================================================================
# Data Loading
# ============================================================================

def load_participant_data(data_dir):
    """Load all participant JSON files (P*.json), sorted chronologically."""
    participants = {}
    json_files = sorted(glob.glob(os.path.join(data_dir, 'P*.json')))

    if not json_files:
        print(f"[WARNING] No JSON files found in {data_dir}")
        return {}

    print(f"[INFO] Found {len(json_files)} JSON files")

    for i, filepath in enumerate(json_files):
        with open(filepath, 'r') as f:
            data = json.load(f)

        pid = data['participant_info']['participant_id']
        total_rounds = len(data.get('rounds', []))
        participants[pid] = {'data': data, 'total_rounds': total_rounds}
        print(f"  [{i + 1:2d}] {pid}: {total_rounds} rounds")

    return {pid: p['data'] for pid, p in participants.items()}


def process_trajectories(rounds):
    """Convert round list → list of trajectory dicts with position arrays & time series."""
    all_trajectories = []
    MAX_FRAME_GAP = 1.0  # cap gaps > 1s (questionnaire pauses)

    for round_data in rounds:
        frames = round_data.get('frames', [])
        if not frames:
            continue

        positions = np.array([f['position'] for f in frames], dtype=float)
        times = [datetime.fromisoformat(f['time']) for f in frames]

        time_series = [0.0]
        for i in range(1, len(times)):
            dt = (times[i] - times[i - 1]).total_seconds()
            if dt > MAX_FRAME_GAP:
                dt = 0.033
            time_series.append(time_series[-1] + dt)

        time_series = np.array(time_series)

        all_trajectories.append({
            'positions': positions,
            'time_series': time_series,
            'task_weight': round_data['task_weight'],
            'target_goal': round_data['target_goal'],
            'duration': float(time_series[-1]),
        })

    return all_trajectories


# ============================================================================
# Per-frame 3D subplot rendering
# ============================================================================

def draw_single_frame_3d(ax, label, trajs, current_time, max_duration):
    """
    Redraw one 3D subplot for the given current_time.
    Freezes (shows completed trajectories, no moving dot) when
    current_time >= max_duration.
    """
    ax.cla()

    # ── Axis limits & camera ─────────────────────────────────────────────────
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)

    # Suppress tick labels to save space in grid
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)

    # Transparent panes & subtle grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#DDDDDD')
    ax.yaxis.pane.set_edgecolor('#DDDDDD')
    ax.zaxis.pane.set_edgecolor('#DDDDDD')
    ax.grid(color='#EEEEEE', linestyle='--', linewidth=0.4, alpha=0.6)

    # ── Static scene elements ────────────────────────────────────────────────

    # Start / Home position
    ax.scatter(*HOME_POSITION, c=START_COLOR, s=55, marker='o',
               edgecolors='white', linewidths=1.5, zorder=5)
    ax.text(HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2] + 28,
            'H', fontsize=5.5, color=START_COLOR, fontweight='bold',
            ha='center', va='bottom')

    # Goal positions
    for goal_idx, goal_pos in GOAL_POSITIONS.items():
        color = GOAL_COLORS[goal_idx]
        ax.scatter(*goal_pos, c=color, s=90, marker='*',
                   edgecolors='white', linewidths=1.0, zorder=5)
        ax.text(goal_pos[0], goal_pos[1], goal_pos[2] + 28,
                f'G{goal_idx}', fontsize=5.5, color=color,
                fontweight='bold', ha='center', va='bottom')

    # Subplot title
    ax.set_title(label, fontsize=8, fontweight='bold', pad=3, color='#333333')

    # ── Trajectory rendering ─────────────────────────────────────────────────
    freeze = current_time >= max_duration

    for traj in trajs:
        positions = traj['positions']
        time_series = traj['time_series']
        color = TASK_WEIGHT_COLORS.get(traj['task_weight'], TRAJECTORY_COLOR)

        if freeze:
            # Show full trajectory, static
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color=color, linewidth=1.0, alpha=0.55, zorder=3)
        else:
            valid_indices = np.where(time_series <= current_time)[0]
            if len(valid_indices) == 0:
                continue

            current_idx = valid_indices[-1]

            # Path drawn so far
            if current_idx > 0:
                ax.plot(positions[:current_idx + 1, 0],
                        positions[:current_idx + 1, 1],
                        positions[:current_idx + 1, 2],
                        color=color, linewidth=1.0, alpha=0.55, zorder=3)

            # Fading trail near the current position
            trail_start = max(0, current_idx - TRAIL_LENGTH)
            trail_pos = positions[trail_start:current_idx + 1]
            if len(trail_pos) > 1:
                for k in range(len(trail_pos) - 1):
                    alpha = 0.25 + 0.75 * (k / max(1, len(trail_pos) - 1))
                    ax.plot(trail_pos[k:k + 2, 0],
                            trail_pos[k:k + 2, 1],
                            trail_pos[k:k + 2, 2],
                            color=color, linewidth=2.8, alpha=alpha, zorder=4)

            # Moving dot (only while this trajectory is still active)
            if current_time < time_series[-1]:
                if current_idx < len(positions) - 1:
                    t1 = time_series[current_idx]
                    t2 = time_series[current_idx + 1]
                    p1 = positions[current_idx]
                    p2 = positions[current_idx + 1]
                    f = np.clip(
                        (current_time - t1) / (t2 - t1) if t2 > t1 else 0,
                        0, 1
                    )
                    cur_pos = p1 + f * (p2 - p1)
                else:
                    cur_pos = positions[current_idx]

                ax.scatter(*cur_pos, c=color, s=70, marker='o',
                           edgecolors='black', linewidths=1.1, zorder=6)


# ============================================================================
# Video rendering (frame-by-frame, no system ffmpeg required)
# ============================================================================

def render_and_save(fig, draw_frame_func, total_frames, output_path, fps, dpi):
    """
    Render each frame to a numpy array and write to MP4.
    Tries imageio-ffmpeg first, falls back to OpenCV.
    """
    # Draw frame 0 first to get canvas size
    draw_frame_func(0)
    fig.canvas.draw()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    print(f"[INFO] Frame size : {canvas_width} × {canvas_height} px")
    print(f"[INFO] Total frames: {total_frames}  ({total_frames / fps:.1f}s video)")

    # ── Attempt 1: imageio + imageio-ffmpeg ──────────────────────────────────
    try:
        import imageio
        import imageio_ffmpeg  # noqa – just verifying it's installed
        print("\n[INFO] Writer: imageio (imageio-ffmpeg)")

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            quality=8,
            macro_block_size=1,     # allows arbitrary frame dimensions
        )

        for frame_num in range(total_frames):
            draw_frame_func(frame_num)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = buf.reshape(canvas_height, canvas_width, 3)
            writer.append_data(frame)

            if frame_num % 15 == 0 or frame_num == total_frames - 1:
                pct = 100.0 * frame_num / total_frames
                print(f"  Frame {frame_num:4d}/{total_frames}  ({pct:5.1f}%)",
                      flush=True)

        writer.close()
        return True

    except ImportError:
        print("[WARNING] imageio-ffmpeg not found. Trying OpenCV...")
    except Exception as e:
        print(f"[WARNING] imageio failed ({e}). Trying OpenCV...")

    # ── Attempt 2: OpenCV ────────────────────────────────────────────────────
    try:
        import cv2
        print("\n[INFO] Writer: OpenCV (mp4v codec)")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (canvas_width, canvas_height))
        if not out.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open output file")

        for frame_num in range(total_frames):
            draw_frame_func(frame_num)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = buf.reshape(canvas_height, canvas_width, 3)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if frame_num % 15 == 0 or frame_num == total_frames - 1:
                pct = 100.0 * frame_num / total_frames
                print(f"  Frame {frame_num:4d}/{total_frames}  ({pct:5.1f}%)",
                      flush=True)

        out.release()
        return True

    except ImportError:
        print("[WARNING] OpenCV not found.")
    except Exception as e:
        print(f"[WARNING] OpenCV failed ({e}).")

    # ── No writer available ──────────────────────────────────────────────────
    print("\n[ERROR] Cannot write MP4 – no suitable writer found.")
    print("        Install one of the following and re-run:")
    print("            pip install imageio[ffmpeg]")
    print("            conda install -c conda-forge opencv")
    return False


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate 3×5 grid 3D trajectory animation video (MP4)'
    )
    parser.add_argument('--input', '-i', type=str,
                        default='./user_study_data',
                        help='Input directory containing P*.json files')
    parser.add_argument('--output', '-o', type=str,
                        default='./trajectory_grid_video_3d.mp4',
                        help='Output MP4 file path')
    parser.add_argument('--dpi', type=int, default=90,
                        help='Rendering DPI (default: 90)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Video frames per second (default: 15)')
    parser.add_argument('--speed', type=float, default=3,
                        help='Animation speed multiplier (default: 3)')
    args = parser.parse_args()

    global FPS, SPEED_MULTIPLIER
    FPS = args.fps
    SPEED_MULTIPLIER = args.speed

    print("\n" + "=" * 70)
    print("  CR5 3D TRAJECTORY GRID VIDEO GENERATOR  (3 rows × 5 cols)")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    participants = load_participant_data(args.input)
    if not participants:
        print("[ERROR] No participant data found!")
        return

    print(f"\n[INFO] Loaded {len(participants)} participants")

    sorted_pids = sorted(participants.keys())   # chronological order

    # ── Process trajectories ─────────────────────────────────────────────────
    participant_data = {}
    for i, pid in enumerate(sorted_pids):
        rounds = participants[pid].get('rounds', [])
        if not rounds:
            print(f"[WARNING] No rounds for {pid}, skipping")
            continue

        trajs = process_trajectories(rounds)
        if not trajs:
            continue

        max_dur = max(t['duration'] for t in trajs)
        label = f'P{i + 1}'
        participant_data[pid] = {
            'trajs': trajs,
            'max_duration': max_dur,
            'label': label,
        }
        print(f"  {label} ({pid}): {len(trajs)} trajectories, "
              f"max = {max_dur:.2f}s")

    pids_ordered = list(participant_data.keys())[:GRID_ROWS * GRID_COLS]
    n_parts = len(pids_ordered)

    global_max_duration = max(
        participant_data[p]['max_duration'] for p in pids_ordered
    )
    animation_duration = global_max_duration / SPEED_MULTIPLIER
    total_frames = int(animation_duration * FPS) + int(FPS * 2)  # +2s freeze at end

    print(f"\n[INFO] Grid layout  : {GRID_ROWS}×{GRID_COLS} ({n_parts} participants)")
    print(f"[INFO] Global max   : {global_max_duration:.2f}s real time")
    print(f"[INFO] Anim duration: {animation_duration:.2f}s at {SPEED_MULTIPLIER}x speed")
    print(f"[INFO] Total frames : {total_frames} @ {FPS} fps")
    print(f"[INFO] Output path  : {os.path.abspath(args.output)}")

    # ── Build figure with 3×5 grid of 3D subplots ────────────────────────────
    # Each 3D cell: 4.5" wide × 4.0" tall  →  figure: 22.5" × 12.0"
    fig_w = GRID_COLS * 4.5
    fig_h = GRID_ROWS * 4.0 + 0.8   # 0.8" for suptitle + legend strip
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white', dpi=args.dpi)

    # Top margin for suptitle; bottom margin for legend
    plt.subplots_adjust(
        left=0.01, right=0.99,
        top=0.92,  bottom=0.08,
        hspace=0.30, wspace=0.08,
    )

    # Create 3D axes
    ax_list = []
    for idx in range(GRID_ROWS * GRID_COLS):
        ax = fig.add_subplot(GRID_ROWS, GRID_COLS, idx + 1, projection='3d')
        ax_list.append(ax)

    # Hide unused slots (in case fewer than 15 participants)
    for idx in range(n_parts, GRID_ROWS * GRID_COLS):
        ax_list[idx].set_visible(False)

    # ── Legend strip (bottom of figure) ─────────────────────────────────────
    legend_items = [
        (TASK_WEIGHT_COLORS[0], 'Task weight = 0  (robot-dominant)'),
        (TASK_WEIGHT_COLORS[1], 'Task weight = 1  (human-dominant)'),
        (GOAL_COLORS[0],        'Goal 0'),
        (GOAL_COLORS[1],        'Goal 1'),
        (START_COLOR,           'Home / Start'),
    ]
    for j, (color, text) in enumerate(legend_items):
        x_pos = 0.04 + j * 0.19
        fig.text(x_pos, 0.025, '●', color=color, fontsize=11,
                 ha='center', va='center', transform=fig.transFigure)
        fig.text(x_pos + 0.005, 0.025, text, fontsize=7.5, color='#444444',
                 ha='left', va='center', transform=fig.transFigure)

    # ── Frame draw function ──────────────────────────────────────────────────
    def draw_frame(frame_num):
        current_time = (frame_num / FPS) * SPEED_MULTIPLIER

        for i, pid in enumerate(pids_ordered):
            d = participant_data[pid]
            draw_single_frame_3d(
                ax_list[i],
                d['label'],
                d['trajs'],
                current_time,
                d['max_duration'],
            )

        # Global title with running time counter
        display_time = min(current_time, global_max_duration)
        fig.suptitle(
            f'CR5 User Study  —  3D End-Effector Trajectories   '
            f'(Speed ×{SPEED_MULTIPLIER}  |  t = {display_time:.1f}s)',
            fontsize=12, fontweight='bold', y=0.975, color='#222222',
        )

    # ── Render ───────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    success = render_and_save(
        fig, draw_frame, total_frames, args.output, FPS, args.dpi
    )
    plt.close(fig)

    if success:
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"\n{'=' * 70}")
        print(f"[DONE]  Video saved → {os.path.abspath(args.output)}")
        print(f"        File size  : {size_mb:.1f} MB")
        print(f"{'=' * 70}\n")
    else:
        print("\n[FAILED] Video was not saved.\n")


if __name__ == '__main__':
    main()
