"""
RFT Results Analysis Tool

Analyzes downloaded RFT evaluation data to show model improvements across training steps.
Analyzes the 10 unique scenarios (each with 5 variations) across RFT steps.
Reads data from analysis_charts/rft_eval/data/ and generates visualizations.

Usage:
    python tools/analyze_rft_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter


class RFTResultsAnalyzer:
    """Analyze RFT evaluation results across training steps."""
    
    def __init__(self):
        """Initialize analyzer with data directory."""
        self.data_dir = Path("analysis_charts/rft_eval/data")
        self.output_dir = Path("analysis_charts/rft_eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.steps_data = {}
        self.steps = []
        
    def load_data(self):
        """Load all step data from JSON files."""
        print("="*70)
        print("📊 RFT RESULTS ANALYSIS")
        print("="*70)
        print()
        print(f"📂 Loading data from {self.data_dir}...\n")
        
        # Find all step files
        step_files = sorted(self.data_dir.glob("Step_*.json"))
        
        if not step_files:
            raise FileNotFoundError(f"No step data files found in {self.data_dir}")
        
        for file in step_files:
            # Extract step number from filename
            step_num = int(file.stem.split("_")[1])
            
            with open(file, 'r') as f:
                data = json.load(f)
            
            print(f"  ✓ Loaded {file.name}: {data['total_items']} items")
            
            self.steps.append(step_num)
            self.steps_data[step_num] = data
        
        self.steps.sort()
        print(f"\n  📈 Found {len(self.steps)} steps: {self.steps}\n")
    
    def extract_scores(self, step):
        """Extract all scores for a given step."""
        items = self.steps_data[step]['items']
        scores = []
        
        for item in items:
            if 'results' in item and item['results']:
                score = item['results'][0].get('score', 0.0)
                scores.append(score)
        
        return scores
    
    def group_by_scenario(self, step):
        """Group items by unique scenario (based on reference_user_message).
        Each of the 50 items is one of 10 scenarios with 5 variations.
        """
        items = self.steps_data[step]['items']
        scenarios = defaultdict(list)
        
        for item in items:
            # Use reference_user_message as scenario identifier
            if 'datasource_item' in item and item['datasource_item']:
                user_msg = item['datasource_item'].get('reference_user_message', '')
                if user_msg:
                    # Get score
                    score = 0.0
                    if 'results' in item and item['results']:
                        score = item['results'][0].get('score', 0.0)
                    
                    scenarios[user_msg].append({
                        'score': score,
                        'item': item
                    })
        
        return scenarios
    
    def analyze_scenarios_across_steps(self):
        """Analyze how each unique scenario performs across steps."""
        print("🔍 Analyzing unique scenarios across steps...\n")
        
        # Get scenarios from Step 0 (baseline)
        step0_scenarios = self.group_by_scenario(self.steps[0])
        scenario_ids = list(step0_scenarios.keys())[:10]  # Get first 10 unique scenarios
        
        print(f"  Found {len(scenario_ids)} unique scenarios\n")
        
        # Track performance of each scenario across steps
        scenario_performance = {}
        
        for scenario_id in scenario_ids:
            scenario_perf = {'steps': {}, 'description': scenario_id[:100] + '...'}
            
            for step in self.steps:
                scenarios = self.group_by_scenario(step)
                if scenario_id in scenarios:
                    scores = [item['score'] for item in scenarios[scenario_id]]
                    scenario_perf['steps'][step] = {
                        'mean': np.mean(scores),
                        'count': len(scores),
                        'scores': scores
                    }
            
            scenario_performance[scenario_id] = scenario_perf
        
        return scenario_performance
    
    def calculate_statistics(self):
        """Calculate statistics for each step."""
        print("📊 Calculating statistics...\n")
        
        stats = {}
        
        for step in self.steps:
            scores = self.extract_scores(step)
            
            # Calculate metrics
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores)
            
            # Calculate pass rate (score >= 0.8)
            pass_count = sum(1 for s in scores if s >= 0.8)
            pass_rate = (pass_count / len(scores)) * 100 if scores else 0
            
            # Score distribution
            perfect_count = sum(1 for s in scores if s == 1.0)
            partial_count = sum(1 for s in scores if 0 < s < 1.0)
            fail_count = sum(1 for s in scores if s == 0.0)
            
            stats[step] = {
                'mean': mean_score,
                'median': median_score,
                'std': std_score,
                'pass_rate': pass_rate,
                'perfect_count': perfect_count,
                'partial_count': partial_count,
                'fail_count': fail_count,
                'total_count': len(scores),
                'scores': scores
            }
            
            print(f"  Step {step}:")
            print(f"    Mean Score: {mean_score:.3f}")
            print(f"    Pass Rate: {pass_rate:.1f}% ({pass_count}/{len(scores)})")
            print(f"    Perfect: {perfect_count}, Partial: {partial_count}, Fail: {fail_count}")
            
            if step > self.steps[0]:
                baseline = stats[self.steps[0]]['mean']
                improvement = ((mean_score - baseline) / baseline) * 100 if baseline > 0 else 0
                print(f"    Improvement from Step 0: {improvement:+.1f}%")
            
            print()
        
        return stats
    
    def plot_score_progression(self, stats):
        """Plot mean score progression across steps."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = self.steps
        means = [stats[s]['mean'] for s in steps]
        
        # Plot line with markers
        ax.plot(steps, means, marker='o', linewidth=2, markersize=10, 
                color='#2E86AB', label='Mean Score')
        
        # Add value labels
        for step, mean in zip(steps, means):
            ax.annotate(f'{mean:.3f}', 
                       xy=(step, mean), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=10,
                       fontweight='bold')
        
        ax.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Improvement Across RFT Steps', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_file = self.output_dir / "score_progression.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_pass_rate_progression(self, stats):
        """Plot pass rate progression across steps."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = self.steps
        pass_rates = [stats[s]['pass_rate'] for s in steps]
        
        # Bar chart with gradient color
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(steps)))
        bars = ax.bar(steps, pass_rates, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, rate in zip(bars, pass_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        ax.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Pass Rate Improvement Across RFT Steps (Score ≥ 0.8)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.output_dir / "pass_rate_progression.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_score_distributions(self, stats):
        """Plot score distributions for each step."""
        # Dynamically determine grid size based on number of steps
        n_steps = len(self.steps)
        n_cols = 3  # Use 3 columns for better layout
        n_rows = (n_steps + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_steps > 1 else [axes]
        
        for idx, step in enumerate(self.steps):
            ax = axes[idx]
            scores = stats[step]['scores']
            
            # Histogram
            ax.hist(scores, bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
            
            # Add mean line
            mean = stats[step]['mean']
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean:.3f}')
            
            ax.set_xlabel('Score', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'Step {step} - Score Distribution', 
                        fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(n_steps, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Score Distributions Across RFT Steps', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.output_dir / "score_distributions.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_outcome_breakdown(self, stats):
        """Plot breakdown of perfect/partial/fail outcomes."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = self.steps
        perfect = [stats[s]['perfect_count'] for s in steps]
        partial = [stats[s]['partial_count'] for s in steps]
        fail = [stats[s]['fail_count'] for s in steps]
        
        x = np.arange(len(steps))
        width = 0.6
        
        # Stacked bar chart
        p1 = ax.bar(x, perfect, width, label='Perfect (1.0)', 
                   color='#06A77D', edgecolor='black')
        p2 = ax.bar(x, partial, width, bottom=perfect, 
                   label='Partial (0.0 < score < 1.0)', 
                   color='#F8B739', edgecolor='black')
        p3 = ax.bar(x, fail, width, bottom=np.array(perfect)+np.array(partial), 
                   label='Fail (0.0)', 
                   color='#D62246', edgecolor='black')
        
        # Add count labels
        for i, step in enumerate(steps):
            total = stats[step]['total_count']
            # Perfect count
            if perfect[i] > 0:
                ax.text(i, perfect[i]/2, str(perfect[i]), 
                       ha='center', va='center', fontweight='bold', fontsize=10)
            # Partial count
            if partial[i] > 0:
                ax.text(i, perfect[i] + partial[i]/2, str(partial[i]), 
                       ha='center', va='center', fontweight='bold', fontsize=10)
            # Fail count
            if fail[i] > 0:
                ax.text(i, perfect[i] + partial[i] + fail[i]/2, str(fail[i]), 
                       ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Items', fontsize=12, fontweight='bold')
        ax.set_title('Outcome Breakdown Across RFT Steps', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Step {s}' for s in steps])
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.output_dir / "outcome_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_improvement_delta(self, stats):
        """Plot improvement delta from baseline (Step 0)."""
        if len(self.steps) < 2:
            print("  ⚠ Skipping delta plot (need at least 2 steps)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        baseline = stats[self.steps[0]]['mean']
        steps = self.steps[1:]  # Skip baseline
        deltas = []
        
        for step in steps:
            delta = ((stats[step]['mean'] - baseline) / baseline) * 100 if baseline > 0 else 0
            deltas.append(delta)
        
        # Bar chart with color based on improvement
        colors = ['#06A77D' if d >= 0 else '#D62246' for d in deltas]
        bars = ax.bar(range(len(steps)), deltas, color=colors, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            height = bar.get_height()
            y_pos = height + 1 if height >= 0 else height - 1
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{delta:+.1f}%',
                   ha='center', va=va,
                   fontsize=11, fontweight='bold')
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement from Baseline (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Improvement Relative to Step 0 (Baseline: {baseline:.3f})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([f'Step {s}' for s in steps])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = self.output_dir / "improvement_delta.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_scenario_heatmap(self, scenario_performance):
        """Plot heatmap showing performance of each scenario across steps."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Prepare data for heatmap
        scenarios = list(scenario_performance.keys())[:10]  # Top 10 scenarios
        steps = self.steps
        
        data = np.zeros((len(scenarios), len(steps)))
        
        for i, scenario_id in enumerate(scenarios):
            for j, step in enumerate(steps):
                if step in scenario_performance[scenario_id]['steps']:
                    data[i, j] = scenario_performance[scenario_id]['steps'][step]['mean']
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(steps)))
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_xticklabels([f'Step {s}' for s in steps])
        
        # Create scenario labels (shortened)
        scenario_labels = []
        for i, scenario_id in enumerate(scenarios):
            # Extract key info from user message
            msg = scenario_id[:60].replace('\n', ' ')
            scenario_labels.append(f"S{i+1}: {msg}...")
        
        ax.set_yticklabels(scenario_labels, fontsize=8)
        
        # Add values to cells
        for i in range(len(scenarios)):
            for j in range(len(steps)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black",
                             fontsize=9, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Score', rotation=270, labelpad=15, fontweight='bold')
        
        ax.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Scenario', fontsize=12, fontweight='bold')
        ax.set_title('Performance Heatmap: Each Scenario Across RFT Steps', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_file = self.output_dir / "scenario_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def plot_scenario_improvement_lines(self, scenario_performance):
        """Plot line chart showing improvement trajectory for each scenario."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios = list(scenario_performance.keys())[:10]  # Top 10 scenarios
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, scenario_id in enumerate(scenarios):
            scenario_data = scenario_performance[scenario_id]
            steps = []
            means = []
            
            for step in self.steps:
                if step in scenario_data['steps']:
                    steps.append(step)
                    means.append(scenario_data['steps'][step]['mean'])
            
            ax.plot(steps, means, marker='o', linewidth=2, markersize=6,
                   color=colors[i], label=f'Scenario {i+1}', alpha=0.7)
        
        ax.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Trajectory by Scenario Across RFT Steps', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', ncol=2)
        
        plt.tight_layout()
        output_file = self.output_dir / "scenario_trajectories.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file}")
    
    def save_summary_report(self, stats):
        """Save detailed summary report as JSON."""
        report = {
            'analysis_date': str(Path.cwd()),
            'steps_analyzed': self.steps,
            'statistics': {}
        }
        
        for step in self.steps:
            report['statistics'][f'Step_{step}'] = {
                'mean_score': float(stats[step]['mean']),
                'median_score': float(stats[step]['median']),
                'std_dev': float(stats[step]['std']),
                'pass_rate': float(stats[step]['pass_rate']),
                'perfect_count': stats[step]['perfect_count'],
                'partial_count': stats[step]['partial_count'],
                'fail_count': stats[step]['fail_count'],
                'total_count': stats[step]['total_count']
            }
        
        # Add improvement metrics
        if len(self.steps) > 1:
            baseline = stats[self.steps[0]]['mean']
            final = stats[self.steps[-1]]['mean']
            total_improvement = ((final - baseline) / baseline) * 100 if baseline > 0 else 0
            
            report['improvement_summary'] = {
                'baseline_step': self.steps[0],
                'baseline_score': float(baseline),
                'final_step': self.steps[-1],
                'final_score': float(final),
                'total_improvement_percent': float(total_improvement)
            }
        
        output_file = self.output_dir / "analysis_summary.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✓ Saved: {output_file}")
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("="*70)
        print("🎯 RFT RESULTS ANALYSIS")
        print("="*70)
        print()
        
        # Load data
        self.load_data()
        
        # Calculate statistics
        stats = self.calculate_statistics()
        
        # Analyze scenarios
        scenario_performance = self.analyze_scenarios_across_steps()
        
        # Generate visualizations
        print("📊 Generating visualizations...\n")
        self.plot_score_progression(stats)
        self.plot_pass_rate_progression(stats)
        self.plot_score_distributions(stats)
        self.plot_outcome_breakdown(stats)
        self.plot_improvement_delta(stats)
        self.plot_scenario_heatmap(scenario_performance)
        self.plot_scenario_improvement_lines(scenario_performance)
        
        # Save summary
        print("\n💾 Saving summary report...\n")
        self.save_summary_report(stats)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"📊 Charts generated: 7")
        print(f"📄 Summary report: analysis_summary.json")
        
        # Print key findings
        if len(self.steps) > 1:
            baseline = stats[self.steps[0]]['mean']
            final = stats[self.steps[-1]]['mean']
            improvement = ((final - baseline) / baseline) * 100 if baseline > 0 else 0
            
            print("\n" + "="*70)
            print("🔍 KEY FINDINGS")
            print("="*70)
            print(f"  Baseline (Step {self.steps[0]}): {baseline:.3f}")
            print(f"  Final (Step {self.steps[-1]}): {final:.3f}")
            print(f"  Total Improvement: {improvement:+.1f}%")
            print()


def main():
    """Main entry point."""
    try:
        analyzer = RFTResultsAnalyzer()
        analyzer.run_analysis()
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Run 'python tools/analyze_rft_eval.py' first to download data")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
