#!/usr/bin/env python3
"""
User Study Data Analysis - Modified Version
Generates single 5-panel figure with duration analysis integrated
Only generates PDF output (no PNG)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr
from pathlib import Path
import os


def setup_plot_style(font_size=20, label_size=22, tick_size=18,
                     line_width=0.8, tick_width=0.8, tick_length=4):
    """Setup matplotlib style for publication-quality figures"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['axes.linewidth'] = line_width
    plt.rcParams['xtick.major.width'] = tick_width
    plt.rcParams['ytick.major.width'] = tick_width
    plt.rcParams['xtick.major.size'] = tick_length
    plt.rcParams['ytick.major.size'] = tick_length
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


# Color scheme matching the original analyze_data.py
TASK_WEIGHT_COLORS = {
    0:  '#E64B35',  # Red
    1: '#00A087',  # Teal
}


def load_json_data(data_dir='user_study_data'):
    """Load all JSON files from user study"""
    data_path = Path(data_dir)
    all_data = []
    
    for json_file in data_path.glob('P*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    
    return all_data


def load_excel_data(excel_dir='user_study_data'):
    excel_path = Path(excel_dir)
    all_dfs = []
    
    for excel_file in excel_path.glob('Scores_P*.xlsx'):
        df = pd.read_excel(excel_file, sheet_name='Data')
        all_dfs.append(df)
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def extract_metrics_from_json(json_data):
    """Extract control effort and duration from JSON files"""
    metrics_list = []
    
    for participant in json_data:
        participant_id = participant['participant_info']['participant_id']
        
        for round_data in participant['rounds']:
            # Calculate control effort
            frames = round_data.get('frames', [])
            if frames:
                user_inputs = np.array([f['user_input'] for f in frames])
                control_effort = float(np.mean(np.linalg.norm(user_inputs, axis=1)))
            else:
                control_effort = round_data.get('avg_user_effort', 0)
            
            metrics_list.append({
                'participant_id': participant_id,
                'round': round_data['round_num'],
                'task_weight': round_data['task_weight'],
                'target_goal': round_data['target_goal'],
                'duration': round_data['duration'],
                'control_effort': control_effort
            })
    
    return pd.DataFrame(metrics_list)


def merge_data(json_metrics_df, excel_df):
    merged = pd.merge(
        json_metrics_df,
        excel_df[['Participant_ID', 'Round', 'Intuitiveness_Score', 'Collaboration_Score']],
        left_on=['participant_id', 'round'],
        right_on=['Participant_ID', 'Round'],
        how='left'
    )
    return merged


def sig_marker(p):
    """Convert p-value to significance marker"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def add_significance_bars(ax, x_positions, values_by_weight, task_weights,
                          pairwise_results, tick_size=18):
    """
    Add significance bars above boxplot.
    Only draws bars for significant pairs (marker != 'ns').
    """
    sig_pairs = [(tw1, tw2) for (tw1, tw2), r in pairwise_results.items()
                 if r['marker'] != 'ns']

    if not sig_pairs:
        return

    all_vals = np.concatenate([values_by_weight[tw] for tw in task_weights])
    y_max_data = all_vals.max()
    y_span = y_max_data

    bar_height = y_span * 0.06
    bar_gap = y_span * 0.10
    base_y = y_max_data + y_span * 0.08

    tw_to_x = {tw: x_positions[i] for i, tw in enumerate(task_weights)}

    for level, (tw1, tw2) in enumerate(sig_pairs):
        marker = pairwise_results[(tw1, tw2)]['marker']
        x1, x2 = tw_to_x[tw1], tw_to_x[tw2]
        y = base_y + level * bar_gap

        # Horizontal line
        ax.plot([x1, x2], [y, y], color='black', linewidth=1.2)
        # Vertical lines
        ax.plot([x1, x1], [y - bar_height * 0.4, y], color='black', linewidth=1.2)
        ax.plot([x2, x2], [y - bar_height * 0.4, y], color='black', linewidth=1.2)
        # Significance marker
        ax.text((x1 + x2) / 2, y + y_span * 0.01, marker,
                ha='center', va='bottom', fontsize=tick_size, color='black')

    # Update y-axis limit to accommodate annotations
    n_levels = len(sig_pairs)
    new_top = base_y + n_levels * bar_gap + y_span * 0.15
    current_bottom = ax.get_ylim()[0]
    ax.set_ylim(current_bottom, new_top)


def create_box_scatter_plot(ax, values_by_weight, ylabel, pairwise_results=None,
                           tick_size=18, label_size=20):
    """
    Create combined box plot + jitter scatter plot in the style of analyze_data.py
    """
    task_weights = sorted(values_by_weight.keys())
    n = len(task_weights)
    colors = [TASK_WEIGHT_COLORS.get(tw, '#999999') for tw in task_weights]
    data_list = [values_by_weight[tw] for tw in task_weights]
    means = [np.mean(v) for v in data_list]
    x = np.arange(n)
    box_width = 0.45
    
    # 1. Box plots
    bp = ax.boxplot(data_list,
                    positions=x,
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2, linestyle='--', color='#555555'),
                    capprops=dict(linewidth=1.2, color='#555555'),
                    boxprops=dict(linewidth=1.2),
                    zorder=2)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    
    # 2. Jitter scatter
    rng = np.random.default_rng(100)
    for i, (tw, color) in enumerate(zip(task_weights, colors)):
        vals = values_by_weight[tw]
        jitter = rng.uniform(-box_width * 0.28, box_width * 0.28, size=len(vals))
        ax.scatter(i + jitter, vals,
                   color=color, edgecolors='white', linewidths=0.4,
                   alpha=0.55, s=28, zorder=3)
    
    # 3. Mean diamond markers
    ax.scatter(x, means,
               marker='D', color='white', edgecolors='black',
               linewidths=1.2, s=55, zorder=5, label='Mean')
    
    # 4. Significance bars (if provided)
    if pairwise_results:
        add_significance_bars(ax, x, values_by_weight, task_weights,
                            pairwise_results, tick_size=tick_size)
    
    # 5. Styling
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_xlabel('Task Weight', fontsize=label_size)
    ax.set_xticks(x)
    # Use simple numeric labels matching core version (e.g., "0", "1")
    ax.set_xticklabels([f'{tw:.1f}' for tw in task_weights], fontsize=tick_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)


def plot_correlation_subplot(ax, merged_df, label_size=20, tick_size=18):
    """
    Scatter plot: Intuitiveness vs Collaboration
    Exactly matching core/analyze_data.py style
    """
    # Extract data
    x = merged_df['Intuitiveness_Score'].values
    y = merged_df['Collaboration_Score'].values
    task_weights_arr = merged_df['task_weight'].values
    unique_weights = sorted(merged_df['task_weight'].unique())
    
    # Scatter plot for each task weight (exactly matching core style)
    for tw in unique_weights:
        mask = task_weights_arr == tw
        color = TASK_WEIGHT_COLORS.get(tw, '#999999')
        ax.scatter(x[mask], y[mask], c=color, s=100, alpha=0.6,
                   edgecolors='black', linewidth=0.8, label=f'Task Weight {tw:.1f}')
    
    # Linear fit (exactly matching core style)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2.5, alpha=0.8, label='Linear fit')
    
    # Calculate and print correlations (keep for console output)
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) > 2:
        r_pearson, p_pearson = pearsonr(x_clean, y_clean)
        r_spearman, p_spearman = spearmanr(x_clean, y_clean)
        print(f"\nIntuitiveness vs Collaboration Correlation:")
        print(f"  Pearson: r={r_pearson:.3f}, p={p_pearson:.4f} {sig_marker(p_pearson)}")
        print(f"  Spearman: ρ={r_spearman:.3f}, p={p_spearman:.4f} {sig_marker(p_spearman)}")
    
    # Axis settings (exactly matching core style)
    ax.set_xlabel('Intuitiveness Score')
    ax.set_ylabel('Collaboration Score')
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, linewidth=1)  # Diagonal reference line
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)


def test_and_get_pairwise_results(values_by_weight, metric_name):
    """
    Perform statistical tests and return pairwise results
    """
    task_weights = sorted(values_by_weight.keys())
    pairwise_results = {}
    
    if len(task_weights) == 2:
        # Mann-Whitney U test for two groups
        try:
            U, p = mannwhitneyu(values_by_weight[task_weights[0]],
                               values_by_weight[task_weights[1]],
                               alternative='two-sided')
            print(f"\n{metric_name} - Mann-Whitney U test:")
            print(f"  {task_weights[0]} vs {task_weights[1]}: U={U:.4f}, p={p:.4f} {sig_marker(p)}")
            pairwise_results[(task_weights[0], task_weights[1])] = {
                'U': U, 'p': p, 'marker': sig_marker(p)
            }
        except ValueError as e:
            print(f"\n{metric_name} - Test failed: {e}")
            pairwise_results[(task_weights[0], task_weights[1])] = {
                'U': 0, 'p': 1.0, 'marker': 'ns'
            }
    
    elif len(task_weights) > 2:
        # Kruskal-Wallis for more than 2 groups
        data_list = [values_by_weight[tw] for tw in task_weights]
        all_values = np.concatenate(data_list)
        
        if len(np.unique(all_values)) > 1:
            try:
                H, p_overall = kruskal(*data_list)
                print(f"\n{metric_name} - Kruskal-Wallis: H={H:.4f}, p={p_overall:.4f} {sig_marker(p_overall)}")
            except ValueError as e:
                print(f"\n{metric_name} - Kruskal-Wallis test failed: {e}")
        else:
            print(f"\n{metric_name} - Kruskal-Wallis test skipped (all values identical)")
        
        # Pairwise comparisons
        for i in range(len(task_weights)):
            for j in range(i + 1, len(task_weights)):
                tw1, tw2 = task_weights[i], task_weights[j]
                try:
                    U, p = mannwhitneyu(values_by_weight[tw1],
                                       values_by_weight[tw2],
                                       alternative='two-sided')
                    n_comp = len(task_weights) * (len(task_weights) - 1) // 2
                    p_corr = min(p * n_comp, 1.0)  # Bonferroni correction
                    print(f"  {tw1} vs {tw2}: U={U:.4f}, p={p:.4f}, p_corr={p_corr:.4f} {sig_marker(p_corr)}")
                    pairwise_results[(tw1, tw2)] = {
                        'U': U, 'p': p_corr, 'marker': sig_marker(p_corr)
                    }
                except ValueError as e:
                    print(f"  {tw1} vs {tw2}: Test failed ({e})")
                    pairwise_results[(tw1, tw2)] = {'U': 0, 'p': 1.0, 'marker': 'ns'}
    
    return pairwise_results


def plot_main_five_panels(merged_df, output_dir='analysis_results',
                         font_size=18, label_size=20, tick_size=16):
    """
    Create the main 5-panel figure (a)(b)(c)(d)(e) in horizontal layout
    Panels a,b,d,e are box plots (same size), panel c is correlation scatter
    """
    setup_plot_style(font_size=font_size, label_size=label_size, tick_size=tick_size)
    
    # 5 subplots with equal widths to match core style
    # Core uses [1,1,1,0.85] for 4 panels, where c is 1
    # CR5 uses [1,1,1,1,1] for 5 panels, all equal width for symmetry
    fig = plt.figure(figsize=(30, 4))
    gs = fig.add_gridspec(1, 5, wspace=0.5, width_ratios=[1, 1, 1, 1, 1])
    
    ax1 = fig.add_subplot(gs[0])  # (a) Intuitiveness
    ax2 = fig.add_subplot(gs[1])  # (b) Collaboration
    ax3 = fig.add_subplot(gs[2])  # (c) Correlation
    ax4 = fig.add_subplot(gs[3])  # (d) User Effort
    ax5 = fig.add_subplot(gs[4])  # (e) Duration (NEW)
    
    # Prepare data by task weight
    task_weights = sorted(merged_df['task_weight'].unique())
    
    # (a) Intuitiveness Score
    intuit_data = {tw: merged_df[merged_df['task_weight'] == tw]['Intuitiveness_Score'].values 
                   for tw in task_weights}
    intuit_results = test_and_get_pairwise_results(intuit_data, 'Intuitiveness Score')
    create_box_scatter_plot(ax1, intuit_data, 'Intuitiveness Score',
                           pairwise_results=intuit_results,
                           tick_size=tick_size, label_size=label_size)
    
    # (b) Collaboration Score
    collab_data = {tw: merged_df[merged_df['task_weight'] == tw]['Collaboration_Score'].values 
                   for tw in task_weights}
    collab_results = test_and_get_pairwise_results(collab_data, 'Collaboration Score')
    create_box_scatter_plot(ax2, collab_data, 'Collaboration Score',
                           pairwise_results=collab_results,
                           tick_size=tick_size, label_size=label_size)
    
    # (c) Intuitiveness vs Collaboration scatter
    plot_correlation_subplot(ax3, merged_df, label_size=label_size, tick_size=tick_size)
    
    # (d) User Effort
    effort_data = {tw: merged_df[merged_df['task_weight'] == tw]['control_effort'].values 
                   for tw in task_weights}
    effort_results = test_and_get_pairwise_results(effort_data, 'User Effort')
    create_box_scatter_plot(ax4, effort_data, 'User Effort',
                           pairwise_results=effort_results,
                           tick_size=tick_size, label_size=label_size)
    
    # (e) Trial Duration (NEW)
    duration_data = {tw: merged_df[merged_df['task_weight'] == tw]['duration'].values 
                     for tw in task_weights}
    duration_results = test_and_get_pairwise_results(duration_data, 'Trial Duration')
    create_box_scatter_plot(ax5, duration_data, 'Trial Duration (s)',
                           pairwise_results=duration_results,
                           tick_size=tick_size, label_size=label_size)
    
    # Add panel labels (a)(b)(c)(d)(e)
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['(a)', '(b)', '(c)', '(d)', '(e)']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=label_size+2, fontweight='bold')
    
    # Save figure (PDF ONLY)
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/main_figure_5panels.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {output_file}")
    plt.close()


def print_summary_statistics(merged_df):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    task_weights = sorted(merged_df['task_weight'].unique())
    
    for tw in task_weights:
        subset = merged_df[merged_df['task_weight'] == tw]
        
        print(f"\nTask Weight = {tw}:")
        print(f"  N trials: {len(subset)}")
        print(f"  Intuitiveness: {subset['Intuitiveness_Score'].mean():.2f} ± {subset['Intuitiveness_Score'].std():.2f}")
        print(f"  Collaboration: {subset['Collaboration_Score'].mean():.2f} ± {subset['Collaboration_Score'].std():.2f}")
        print(f"  Control Effort: {subset['control_effort'].mean():.2f} ± {subset['control_effort'].std():.2f}")
        print(f"  Duration: {subset['duration'].mean():.2f} ± {subset['duration'].std():.2f} seconds")


def main():
    print("="*70)
    print("USER STUDY DATA ANALYSIS - MODIFIED VERSION")
    print("="*70)
    
    print("\nLoading data...")
    json_data = load_json_data('user_study_data')
    excel_df = load_excel_data('user_study_data')
    
    if len(json_data) == 0:
        print("Error: No JSON files found")
        return
    
    if excel_df.empty:
        print("Error: No Excel files found. Please fill questionnaire scores.")
        return
    
    print(f"Loaded {len(json_data)} participants")
    print(f"Loaded {len(excel_df)} questionnaire responses")
    
    json_metrics_df = extract_metrics_from_json(json_data)
    merged_df = merge_data(json_metrics_df, excel_df)
    merged_df = merged_df.dropna(subset=['Intuitiveness_Score', 'Collaboration_Score'])
    
    print(f"Final dataset: {len(merged_df)} complete trials")
    
    print_summary_statistics(merged_df)
    
    print("\nGenerating figure...")
    plot_main_five_panels(merged_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated:")
    print("  - analysis_results/main_figure_5panels.pdf")


if __name__ == "__main__":
    main()