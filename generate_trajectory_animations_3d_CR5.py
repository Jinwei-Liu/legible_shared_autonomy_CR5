#!/usr/bin/env python3
"""
Generate 3D animated GIF visualizations for CR5 user study trajectories
Beautiful 3D rendering with multiple views and smooth animations
"""

import os
import json
import glob
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle, FancyBboxPatch
import mpl_toolkits.mplot3d.art3d as art3d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration - From CR5 config.py and user_study_system.py
# ============================================================================

# Workspace boundaries
X_MIN, X_MAX = 300, 700
Y_MIN, Y_MAX = -200, 150
Z_MIN, Z_MAX = 320, 600

# Goal positions (from user_study_system.py line 25-28)
GOAL_POSITIONS = {
    0: np.array([650, 0, 350]),     # Goal 0
    1: np.array([700, -50, 350])    # Goal 1
}

# Home/Start position
HOME_POSITION = np.array([300.0, -25.0, 500.0])

# Appearance settings
GOAL_RADIUS = 20
ROBOT_SIZE = 8

# Color scheme - Professional and elegant
GOAL_COLORS = {
    0: '#3C5488',  # Deep Blue
    1: '#F39B7F'   # Warm Salmon
}

TASK_WEIGHT_COLORS = {
    0:  '#E64B35',  # Red
    1: '#00A087',  # Teal
}
TRAJECTORY_COLOR = '#7E6148'  # Brown
START_COLOR = '#00A087'       # Teal
WORKSPACE_COLOR = '#E0E0E0'   # Light gray
GRID_COLOR = '#CCCCCC'        # Gray

# Animation parameters
FPS = 15                      # Frames per second
TRAIL_LENGTH = 20             # Number of recent positions to show in trail
SPEED_MULTIPLIER = 3          # Animation speed (3x real-time)
DPI = 120                     # High quality output

# Camera settings for beautiful 3D view
CAMERA_ELEVATION = 25         # Degrees above horizon
CAMERA_AZIMUTH = 45          # Degrees around z-axis
CAMERA_DISTANCE = 1.4        # Zoom level


# ============================================================================
# Data Loading
# ============================================================================

def load_participant_data(data_dir):
    """
    Load all participant experiment data from CR5 user study
    Returns: dict with participant_id as key
    """
    participants = {}
    
    json_files = glob.glob(os.path.join(data_dir, 'P*.json'))
    
    if not json_files:
        print(f"[WARNING] No JSON files found in {data_dir}")
        return {}
    
    print(f"[INFO] Found {len(json_files)} JSON files")
    
    for filepath in sorted(json_files):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            pid = data['participant_info']['participant_id']
            total_rounds = len(data['rounds'])
            
            participants[pid] = {
                'data': data,
                'total_rounds': total_rounds
            }
            print(f"  Participant {pid}: {total_rounds} rounds")
    
    return {pid: p['data'] for pid, p in participants.items()}


def extract_rounds(exp_data):
    """Extract rounds from experiment data"""
    return exp_data.get('rounds', [])


# ============================================================================
# 3D Visualization Helpers
# ============================================================================

def draw_workspace_box(ax):
    """Draw a beautiful semi-transparent workspace boundary box"""
    # Define box vertices
    vertices = [
        [X_MIN, Y_MIN, Z_MIN],
        [X_MAX, Y_MIN, Z_MIN],
        [X_MAX, Y_MAX, Z_MIN],
        [X_MIN, Y_MAX, Z_MIN],
        [X_MIN, Y_MIN, Z_MAX],
        [X_MAX, Y_MIN, Z_MAX],
        [X_MAX, Y_MAX, Z_MAX],
        [X_MIN, Y_MAX, Z_MAX]
    ]
    
    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # Left
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # Top
    ]
    
    # Draw faces
    face_collection = Poly3DCollection(faces, alpha=0.05, 
                                       facecolor=WORKSPACE_COLOR,
                                       edgecolor=GRID_COLOR, linewidths=1)
    ax.add_collection3d(face_collection)
    
    # Draw grid on bottom face
    x_grid = np.linspace(X_MIN, X_MAX, 5)
    y_grid = np.linspace(Y_MIN, Y_MAX, 5)
    
    for x in x_grid:
        ax.plot([x, x], [Y_MIN, Y_MAX], [Z_MIN, Z_MIN], 
               color=GRID_COLOR, linewidth=0.5, alpha=0.3)
    for y in y_grid:
        ax.plot([X_MIN, X_MAX], [y, y], [Z_MIN, Z_MIN], 
               color=GRID_COLOR, linewidth=0.5, alpha=0.3)


def draw_goal_spheres(ax):
    """Draw goal positions as beautiful spheres"""
    for goal_idx, goal_pos in GOAL_POSITIONS.items():
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = GOAL_RADIUS * np.outer(np.cos(u), np.sin(v)) + goal_pos[0]
        y = GOAL_RADIUS * np.outer(np.sin(u), np.sin(v)) + goal_pos[1]
        z = GOAL_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v)) + goal_pos[2]
        
        color = GOAL_COLORS[goal_idx]
        ax.plot_surface(x, y, z, color=color, alpha=0.4, shade=True,
               edgecolors=(0,0,0,0), linewidth=0)
        # Draw goal center point
        ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], 
                  c=color, s=100, marker='o', edgecolors='white', linewidths=2)
        
        # Add label
        ax.text(goal_pos[0], goal_pos[1], goal_pos[2] + 40, 
               f'Goal {goal_idx}',
               fontsize=10, fontweight='bold', color=color,
               ha='center', va='bottom')


def draw_start_position(ax):
    """Draw home/start position"""
    # Start sphere
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 8)
    x = ROBOT_SIZE * 0.8 * np.outer(np.cos(u), np.sin(v)) + HOME_POSITION[0]
    y = ROBOT_SIZE * 0.8 * np.outer(np.sin(u), np.sin(v)) + HOME_POSITION[1]
    z = ROBOT_SIZE * 0.8 * np.outer(np.ones(np.size(u)), np.cos(v)) + HOME_POSITION[2]
    
    ax.plot_surface(x, y, z, color=START_COLOR, alpha=0.3, shade=True,
               edgecolors=(0,0,0,0), linewidth=0)
    ax.scatter([HOME_POSITION[0]], [HOME_POSITION[1]], [HOME_POSITION[2]], 
              c=START_COLOR, s=80, marker='o', edgecolors='white', linewidths=1.5)
    
    # Add label
    ax.text(HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2] + 40, 
           'Start',
           fontsize=9, fontweight='bold', color=START_COLOR,
           ha='center', va='bottom')


def draw_robot_at_position(ax, position, color, alpha=1.0):
    """Draw robot (end-effector) at given position"""
    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 8)
    x = ROBOT_SIZE * np.outer(np.cos(u), np.sin(v)) + position[0]
    y = ROBOT_SIZE * np.outer(np.sin(u), np.sin(v)) + position[1]
    z = ROBOT_SIZE * np.outer(np.ones(np.size(u)), np.cos(v)) + position[2]
    
    ax.plot_surface(x, y, z, color=color, alpha=0.55,
               shade=True,
               edgecolors=(0,0,0,0), linewidth=0)


# ============================================================================
# Animation Generation
# ============================================================================

def create_3d_trajectory_animation(participant_id, rounds, output_dir, 
                                   dpi=DPI, speed_multiplier=SPEED_MULTIPLIER):
    """
    Create beautiful 3D animated GIF showing all trajectories for one participant
    
    Args:
        participant_id: Participant identifier
        rounds: List of round data
        output_dir: Directory to save output GIF
        dpi: Resolution of the output
        speed_multiplier: Animation speed multiplier (e.g., 3 means 3x real-time speed)
    """
    
    print(f"\n[Processing] Participant {participant_id}")
    print(f"  Total rounds: {len(rounds)}")
    
    # Set up the figure with 3D axis
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='#F8F8F8')
    
    # Collect all trajectories with timestamps
    all_trajectories = []
    for round_data in rounds:
        frames = round_data['frames']
        positions = np.array([f['position'] for f in frames])
        
        # Extract timestamps and compute cumulative time
        times = [datetime.fromisoformat(f['time']) for f in frames]
        
        # Compute time differences (cap large gaps from pauses)
        time_series = [0.0]
        MAX_FRAME_GAP = 1.0
        
        for i in range(1, len(times)):
            dt = (times[i] - times[i-1]).total_seconds()
            if dt > MAX_FRAME_GAP:
                dt = 0.033  # ~30 FPS
            time_series.append(time_series[-1] + dt)
        
        time_series = np.array(time_series)
        
        task_weight = round_data['task_weight']
        target_goal_idx = round_data['target_goal']
        
        all_trajectories.append({
            'positions': positions,
            'time_series': time_series,
            'task_weight': task_weight,
            'target_goal_idx': target_goal_idx,
            'round_num': round_data['round_num'],
            'duration': time_series[-1] if len(time_series) > 0 else 0
        })
    
    print(f"  Extracted {len(all_trajectories)} trajectories")
    
    # Calculate animation parameters
    max_duration = max(traj['duration'] for traj in all_trajectories)
    animation_duration = max_duration / speed_multiplier
    total_frames = int(animation_duration * FPS) + 30  # Extra frames at end
    
    print(f"  Max trajectory duration: {max_duration:.2f}s")
    print(f"  Animation duration at {speed_multiplier}x speed: {animation_duration:.2f}s")
    print(f"  Generating {total_frames} frames at {FPS} fps...")
    
    def init():
        """Initialize animation"""
        ax.clear()
        
        # Set workspace limits
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_zlim(Z_MIN, Z_MAX)
        
        # Set camera view
        ax.view_init(elev=CAMERA_ELEVATION, azim=CAMERA_AZIMUTH)
        ax.dist = CAMERA_DISTANCE
        
        # Labels
        ax.set_xlabel('X (mm)', fontsize=10, labelpad=10)
        ax.set_ylabel('Y (mm)', fontsize=10, labelpad=10)
        ax.set_zlabel('Z (mm)', fontsize=10, labelpad=10)
        
        # Draw environment
        draw_workspace_box(ax)
        draw_start_position(ax)
        draw_goal_spheres(ax)
        
        # Title
        ax.set_title(f'Participant {participant_id} - 3D Trajectories',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Make panes transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Lighten grid
        ax.xaxis.pane.set_edgecolor('#DDDDDD')
        ax.yaxis.pane.set_edgecolor('#DDDDDD')
        ax.zaxis.pane.set_edgecolor('#DDDDDD')
        ax.grid(color='#DDDDDD', linestyle='--', linewidth=0.5, alpha=0.5)
        
        return []
    
    def animate(frame_num):
        """Animation function - all trajectories move simultaneously"""
        
        # Don't clear everything, just remove trajectory elements
        # Keep workspace, goals, and start position
        
        # Calculate current real time
        current_time = (frame_num / FPS) * speed_multiplier
        
        # Draw all trajectories up to current time
        for traj in all_trajectories:
            positions = traj['positions']
            time_series = traj['time_series']
            task_weight = traj['task_weight']
            color = TASK_WEIGHT_COLORS.get(task_weight, TRAJECTORY_COLOR)
            
            # Find which frames to display
            valid_indices = np.where(time_series <= current_time)[0]
            
            if len(valid_indices) > 0:
                current_idx = valid_indices[-1]
                
                # Draw completed path
                if current_idx > 0:
                    ax.plot(positions[:current_idx+1, 0], 
                           positions[:current_idx+1, 1], 
                           positions[:current_idx+1, 2],
                           color=color, linewidth=2.5, alpha=0.7, zorder=3)
                
                # Draw trail with fade effect
                trail_start = max(0, current_idx - TRAIL_LENGTH)
                trail_positions = positions[trail_start:current_idx+1]
                
                if len(trail_positions) > 1:
                    for i in range(len(trail_positions) - 1):
                        alpha = 0.4 + 0.6 * (i / max(1, len(trail_positions) - 1))
                        ax.plot(trail_positions[i:i+2, 0],
                               trail_positions[i:i+2, 1],
                               trail_positions[i:i+2, 2],
                               color=color, linewidth=4, alpha=alpha, zorder=4)
                
                # Draw current robot position if trajectory is still active
                if current_time < time_series[-1]:
                    # Interpolate position if between frames
                    if current_idx < len(positions) - 1:
                        t1, t2 = time_series[current_idx], time_series[current_idx + 1]
                        p1, p2 = positions[current_idx], positions[current_idx + 1]
                        
                        alpha_interp = (current_time - t1) / (t2 - t1) if t2 > t1 else 0
                        alpha_interp = np.clip(alpha_interp, 0, 1)
                        
                        current_pos = p1 + alpha_interp * (p2 - p1)
                    else:
                        current_pos = positions[current_idx]
                    
                    # Draw robot at current position
                    draw_robot_at_position(ax, current_pos, color, alpha=0.9)
        
        # Speed indicator (top-left in 2D overlay)
        ax.text2D(0.05, 0.95, f'Speed: ×{speed_multiplier}',
                 transform=ax.transAxes, fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Time indicator
        ax.text2D(0.05, 0.05, f'Time: {current_time:.2f}s',
                 transform=ax.transAxes, fontsize=10, color='gray',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Rotate camera slightly for dynamic effect
        ax.view_init(elev=CAMERA_ELEVATION, azim=CAMERA_AZIMUTH + frame_num * 0.1)
        
        return []
    
    # Create animation
    init()
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=total_frames, interval=1000/FPS,
                                  blit=False, repeat=True)
    
    # Save as GIF
    output_path = os.path.join(output_dir, f'participant_{participant_id}_3D_trajectories.gif')
    
    print(f"  Saving animation to {output_path}...")
    writer = animation.PillowWriter(fps=FPS)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    
    print(f"  [SAVED] {output_path}")
    
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    global FPS, SPEED_MULTIPLIER, DPI

    parser = argparse.ArgumentParser(
        description='Generate beautiful 3D animated GIF visualizations for CR5 user study'
    )
    parser.add_argument('--input', '-i', type=str, 
                       default='./user_study_data',
                       help='Input directory containing JSON files')
    parser.add_argument('--output', '-o', type=str, 
                       default='./trajectory_animations_3d',
                       help='Output directory for GIF animations')
    parser.add_argument('--dpi', type=int, default=DPI,
                       help=f'DPI for output GIFs (default: {DPI})')
    parser.add_argument('--fps', type=int, default=FPS,
                       help=f'Frames per second (default: {FPS})')
    parser.add_argument('--speed', type=float, default=SPEED_MULTIPLIER,
                       help=f'Animation speed multiplier (default: {SPEED_MULTIPLIER})')
    
    args = parser.parse_args()

    FPS = args.fps
    SPEED_MULTIPLIER = args.speed
    DPI = args.dpi
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load all participant data
    print("\n" + "="*70)
    print("  CR5 3D TRAJECTORY ANIMATION GENERATOR")
    print("="*70)
    
    participants = load_participant_data(args.input)
    
    if not participants:
        print("[ERROR] No participant data found!")
        return
    
    print(f"\n[INFO] Loaded {len(participants)} participant(s)")
    
    # Generate animation for each participant
    for participant_id, exp_data in participants.items():
        rounds = extract_rounds(exp_data)
        
        if not rounds:
            print(f"[WARNING] No rounds found for participant {participant_id}, skipping")
            continue
        
        create_3d_trajectory_animation(participant_id, rounds, args.output, 
                                      dpi=args.dpi, speed_multiplier=args.speed)
    
    print("\n" + "="*70)
    print(f"[DONE] All 3D animations saved to: {args.output}/")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
