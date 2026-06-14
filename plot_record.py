#!/usr/bin/env python3
"""
plot_record.py  ─  Visualize a single test3_record session as an animated GIF

Layout:
  Left  (~65 %): 3D animated trajectory
                 - single colour path
                 - red  quiver arrows = human input  (direction & magnitude)
                 - green quiver arrows = robot action (direction & magnitude)
                   historical arrows (faded) every N frames + bright at head
                 - elapsed time overlay
  Right (~27 %): Horizontal belief bar chart, updated each frame

Usage:
  python plot_record.py trajectory_Standard_20260228_123456.json
  python plot_record.py trajectory_Legible_20260228_123456.json -o viz.gif
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')                         # off-screen backend for GIF saving
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401 – needed for 3-D projection
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')


# ── workspace boundaries (from config.py) ─────────────────────────────────────
X_MIN, X_MAX = 300, 700
Y_MIN, Y_MAX = -200, 150
Z_MIN, Z_MAX = 320, 600

# ── colours ───────────────────────────────────────────────────────────────────
GOAL_PALETTE = ['#3C5488', '#F39B7F', '#00A087', '#E64B35']  # one per goal
TRAJ_COLOR   = '#4393C3'   # single trajectory colour (steel blue)
START_COLOR  = '#00A087'   # start / home marker
HUMAN_COLOR  = '#D62728'   # red   – human input arrows
ROBOT_COLOR  = '#2CA02C'   # green – robot action arrows

# ── arrow display ─────────────────────────────────────────────────────────────
# CONTROL_SPEED is ~50 mm/s, so ARROW_SCALE=2.5 → max ~125 mm (≈31 % of X span)
ARROW_SCALE  = 2.5     # mm/s  → mm display length
ARROW_SKIP   = 8       # show a historical arrow every N data-frames
HIST_ALPHA   = 0.22    # opacity of historical arrows
HEAD_ALPHA   = 0.90    # opacity of current-position arrows
MIN_VEC_MM   = 0.5     # skip arrow if scaled length < this (mm)

# ── geometry ──────────────────────────────────────────────────────────────────
GOAL_R = 18   # goal-sphere radius (mm)
BOT_R  =  7   # robot-head sphere radius (mm)

# ── output ────────────────────────────────────────────────────────────────────
FPS         = 15
CAMERA_ELEV = 28
CAMERA_AZIM = 45


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sphere_mesh(cx, cy, cz, r, nu=16, nv=8):
    """Return (X, Y, Z) surface mesh for a sphere centred at (cx, cy, cz)."""
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi,     nv)
    x = r * np.outer(np.cos(u), np.sin(v)) + cx
    y = r * np.outer(np.sin(u), np.sin(v)) + cy
    z = r * np.outer(np.ones_like(u), np.cos(v)) + cz
    return x, y, z


def _setup_3d_scene(ax, goals, home):
    """Draw the static scene: axes, workspace grid, start, and goal spheres."""
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)

    ax.set_xlabel('X (mm)', labelpad=8, fontsize=9)
    ax.set_ylabel('Y (mm)', labelpad=8, fontsize=9)
    ax.set_zlabel('Z (mm)', labelpad=8, fontsize=9)
    ax.tick_params(labelsize=7)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#DDDDDD')
    ax.yaxis.pane.set_edgecolor('#DDDDDD')
    ax.zaxis.pane.set_edgecolor('#DDDDDD')
    ax.grid(color='#DDDDDD', linestyle='--', linewidth=0.4, alpha=0.5)

    # start / home marker
    ax.plot_surface(*_sphere_mesh(*home, BOT_R * 0.9),
                    color=START_COLOR, alpha=0.50,
                    edgecolors=(0, 0, 0, 0), linewidth=0, shade=True)
    ax.text(home[0], home[1], home[2] + 32,
            'Start', fontsize=8, color=START_COLOR,
            ha='center', va='bottom', fontweight='bold')

    # goal spheres
    for i, g in enumerate(goals):
        c = GOAL_PALETTE[i % len(GOAL_PALETTE)]
        ax.plot_surface(*_sphere_mesh(*g, GOAL_R),
                        color=c, alpha=0.32,
                        edgecolors=(0, 0, 0, 0), linewidth=0, shade=True)
        ax.text(g[0], g[1], g[2] + 38,
                f'Goal {i + 1}', fontsize=9, color=c,
                ha='center', va='bottom', fontweight='bold')


def _quiver(ax, pos, vec_raw, color, alpha, scale=ARROW_SCALE):
    """Draw a single quiver arrow. Skip if too short."""
    v = np.asarray(vec_raw, dtype=float) * scale
    if np.linalg.norm(v) < MIN_VEC_MM:
        return
    ax.quiver(pos[0], pos[1], pos[2],
              v[0],   v[1],   v[2],
              color=color, alpha=alpha,
              linewidth=1.8, arrow_length_ratio=0.28)


# ─────────────────────────────────────────────────────────────────────────────
# Main visualiser
# ─────────────────────────────────────────────────────────────────────────────

def create_animation(data: dict, output_path: str) -> None:
    frames   = data['frames']
    goals    = np.array(data['goals'])
    home     = np.array(data['home_position'])
    n_goals  = len(goals)
    N        = len(frames)

    positions  = np.array([f['position']     for f in frames])   # (N, 3)
    beliefs    = np.array([f['beliefs']      for f in frames])   # (N, n_goals)
    user_in    = np.array([f['user_input']   for f in frames])   # (N, 3)
    robot_act  = np.array([f['robot_action'] for f in frames])   # (N, 3)
    times      = np.array([f['time']         for f in frames])   # (N,)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    # 3-D panel (left 64 %)
    ax3 = fig.add_axes([0.01, 0.04, 0.62, 0.92],
                       projection='3d', facecolor='#F8F8F8')
    # belief bar chart (right 27 %)
    ax_b = fig.add_axes([0.70, 0.20, 0.27, 0.58])

    # ── static belief-bar setup ───────────────────────────────────────────────
    bar_colors = [GOAL_PALETTE[i % len(GOAL_PALETTE)] for i in range(n_goals)]
    goal_labels = [f'Goal {i + 1}' for i in range(n_goals)]

    bars = ax_b.barh(range(n_goals), beliefs[0],
                     color=bar_colors, edgecolor='white', height=0.55)
    ax_b.set_xlim(0, 1)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xticklabels(['0', '.25', '.5', '.75', '1'], fontsize=8)
    ax_b.set_yticks(range(n_goals))
    ax_b.set_yticklabels(goal_labels, fontsize=10)
    ax_b.invert_yaxis()
    ax_b.set_xlabel('Belief probability', fontsize=9)
    ax_b.set_title('Belief Update', fontsize=11, fontweight='bold', pad=8)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # text labels on bars
    bel_texts = []
    for i in range(n_goals):
        t = ax_b.text(beliefs[0][i] + 0.02, i,
                      f'{beliefs[0][i]:.2f}',
                      va='center', ha='left', fontsize=9)
        bel_texts.append(t)

    # time readout below bar chart
    time_label_bar = ax_b.text(0.5, -0.22, 'Time: 0.00 s',
                                transform=ax_b.transAxes,
                                ha='center', fontsize=9, color='#555555')

    # ── legend handles (created once, reused every frame) ─────────────────────
    legend_handles = [
        Line2D([0], [0], color=TRAJ_COLOR,  linewidth=2.5, label='Trajectory'),
        Line2D([0], [0], color=HUMAN_COLOR, linewidth=2.5, label='Human input'),
        Line2D([0], [0], color=ROBOT_COLOR, linewidth=2.5, label='Robot action'),
    ]

    # ── animation ─────────────────────────────────────────────────────────────
    # Target real-time playback; interval in ms
    avg_dt   = (times[-1] - times[0]) / max(N - 1, 1)
    interval = max(avg_dt * 1000, 1000.0 / FPS)

    def animate(fi: int):
        # ── 3-D panel: clear and fully redraw ────────────────────────────────
        ax3.cla()
        _setup_3d_scene(ax3, goals, home)

        pos = positions[fi]

        # completed path
        if fi > 0:
            ax3.plot(positions[:fi + 1, 0],
                     positions[:fi + 1, 1],
                     positions[:fi + 1, 2],
                     color=TRAJ_COLOR, linewidth=2.2, alpha=0.80)

        # historical arrows (faded) along the completed path
        for j in range(0, fi, ARROW_SKIP):
            _quiver(ax3, positions[j], user_in[j],   HUMAN_COLOR, HIST_ALPHA)
            _quiver(ax3, positions[j], robot_act[j], ROBOT_COLOR, HIST_ALPHA)

        # current arrows (bright) at robot head
        _quiver(ax3, pos, user_in[fi],   HUMAN_COLOR, HEAD_ALPHA)
        _quiver(ax3, pos, robot_act[fi], ROBOT_COLOR, HEAD_ALPHA)

        # robot sphere at head
        ax3.plot_surface(*_sphere_mesh(*pos, BOT_R),
                         color=TRAJ_COLOR, alpha=0.80,
                         edgecolors=(0, 0, 0, 0), linewidth=0, shade=True)

        # elapsed time overlay (bottom-left of 3-D panel)
        ax3.text2D(0.04, 0.04,
                   f'Time: {times[fi]:.2f} s',
                   transform=ax3.transAxes,
                   fontsize=10, color='#333333',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

        # legend (top-left of 3-D panel)
        ax3.legend(handles=legend_handles,
                   loc='upper left', fontsize=8,
                   framealpha=0.85, handlelength=1.6)

        # ── belief bar chart: update widths and text ──────────────────────────
        bel = beliefs[fi]
        for i, bar in enumerate(bars):
            bar.set_width(bel[i])
        for i, txt in enumerate(bel_texts):
            x_val = min(bel[i] + 0.02, 0.80)  # keep text inside axes
            txt.set_x(x_val)
            txt.set_text(f'{bel[i]:.2f}')
        time_label_bar.set_text(f'Time: {times[fi]:.2f} s')

        return []

    anim = animation.FuncAnimation(
        fig, animate,
        frames=N, interval=interval,
        blit=False, repeat=True
    )

    writer = animation.PillowWriter(fps=FPS)
    print(f'Saving  →  {output_path}')
    print(f'  Data frames : {N}')
    print(f'  Interval    : {interval:.1f} ms  (real-time playback)')
    print(f'  GIF fps     : {FPS}')
    anim.save(output_path, writer=writer, dpi=100)
    plt.close(fig)
    print('Done.')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Animate a single trajectory recorded by test3_record.py'
    )
    parser.add_argument(
        'input', default='./trajectory_Standard_20260228_143642.json',
        help='Path to the JSON file produced by test3_record'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output GIF path (default: <input stem>_viz.gif)'
    )
    args = parser.parse_args()

    # default output name
    out = args.output
    if out is None:
        stem = args.input.rsplit('.', 1)[0] if '.' in args.input else args.input
        out  = stem + '_viz.gif'

    with open(args.input, 'r') as fh:
        data = json.load(fh)

    print(f'Loaded  : {args.input}')
    print(f'Mode    : {data.get("autonomy_mode", "?")}')
    print(f'Goals   : {len(data["goals"])}')
    print(f'Frames  : {len(data["frames"])}')
    if data['frames']:
        print(f'Duration: {data["frames"][-1]["time"]:.2f} s')

    create_animation(data, out)


if __name__ == '__main__':
    main()
