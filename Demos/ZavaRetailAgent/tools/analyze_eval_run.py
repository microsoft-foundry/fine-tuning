"""
Evaluation Run Analysis Tool

Downloads and analyzes Azure OpenAI evaluation runs, generating comprehensive
statistics, comparisons, and visualizations.

Usage:
    python tools/analyze_eval_run.py RESOURCE_NAME EVAL_ID
    python tools/analyze_eval_run.py omi-fdp-swc-resource eval_691abff78ca88191ad5c8ffd039a40d1
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EvaluationAnalyzer:
    """Analyze Azure OpenAI evaluation runs."""
    
    def __init__(self, eval_id, resource_name="omi-fdp-swc-resource"):
        """Initialize analyzer with evaluation ID."""
        self.eval_id = eval_id
        self.resource_name = resource_name
        self.output_dir = Path("analysis_charts/eval_run")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Azure OpenAI client
        print(f"🔧 Initializing Azure OpenAI client...")
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        
        self.client = AzureOpenAI(
            azure_endpoint=f"https://{resource_name}.openai.azure.com",
            azure_ad_token_provider=token_provider,
            api_version="2025-04-01-preview"
        )
        print(f"✓ Client initialized\n")
        
        self.evaluation = None
        self.runs = []
        self.run_results = {}
    
    def fetch_evaluation(self):
        """Fetch evaluation details."""
        print(f"📥 Fetching evaluation: {self.eval_id}")
        try:
            self.evaluation = self.client.evals.retrieve(self.eval_id)
            print(f"✓ Evaluation: {self.evaluation.name}")
            print(f"  ID: {self.evaluation.id}")
            if hasattr(self.evaluation, 'created_at'):
                print(f"  Created: {self.evaluation.created_at}")
            print()
        except Exception as e:
            print(f"❌ Error fetching evaluation: {e}")
            raise
    
    def fetch_runs(self):
        """Fetch all runs for this evaluation."""
        print(f"📥 Fetching evaluation runs...")
        try:
            runs_page = self.client.evals.runs.list(eval_id=self.eval_id)
            self.runs = list(runs_page)
            print(f"✓ Found {len(self.runs)} runs:")
            for run in self.runs:
                print(f"  - {run.name} (ID: {run.id})")
                if hasattr(run, 'status'):
                    print(f"    Status: {run.status}")
            print()
        except Exception as e:
            print(f"❌ Error fetching runs: {e}")
            raise
    
    def fetch_run_results(self):
        """Fetch detailed results for each run."""
        print(f"📥 Fetching results for each run...")
        
        for run in self.runs:
            print(f"  Fetching results for {run.name}...")
            try:
                output_items = self.client.evals.runs.output_items.list(
                    eval_id=self.eval_id,
                    run_id=run.id
                )
                
                items = list(output_items)
                self.run_results[run.name] = {
                    'run': run,
                    'items': items,
                    'total_items': len(items)
                }
                print(f"    ✓ Retrieved {len(items)} items")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.run_results[run.name] = {
                    'run': run,
                    'items': [],
                    'total_items': 0,
                    'error': str(e)
                }
        print()
    
    def analyze_results(self):
        """Analyze results and compute statistics."""
        print(f"📊 Analyzing results...\n")
        
        analysis = {}
        
        for model_name, data in self.run_results.items():
            items = data['items']
            
            if not items:
                print(f"⚠️  No items for {model_name}")
                continue
            
            scores = []
            error_counts = defaultdict(int)
            pass_count = 0
            score_ranges = defaultdict(int)  # Track score distribution
            sample_errors = []  # Store sample errors for detailed analysis
            
            for item in items:
                # Extract score from results
                if hasattr(item, 'results') and item.results:
                    for result in item.results:
                        if hasattr(result, 'score') and result.score is not None:
                            score = result.score
                            scores.append(score)
                            
                            # Track score ranges for histogram
                            score_bucket = int(score * 10) / 10
                            score_ranges[score_bucket] += 1
                            
                            # Categorize by score
                            if score >= 1.0:
                                pass_count += 1
                            elif score >= 0.5:
                                error_counts['ARGUMENT_VALUES_MISMATCH'] += 1
                                if len(sample_errors) < 10:
                                    sample_errors.append({'score': score, 'type': 'ARGUMENT_VALUES_MISMATCH'})
                            elif score >= 0.4:
                                error_counts['ARGUMENTS_MISMATCH'] += 1
                            elif score >= 0.3:
                                error_counts['NAME_MISMATCH'] += 1
                            elif score >= 0.22:
                                error_counts['MISSING_FUNCTION'] += 1
                            elif score >= 0.21:
                                error_counts['NO_FUNCTION'] += 1
                            elif score >= 0.1:
                                error_counts['ARGUMENT_JSON_PARSE_ERROR'] += 1
                            else:
                                error_counts['UNKNOWN_ERROR'] += 1
            
            total_samples = len(scores)
            pass_rate = pass_count / total_samples if total_samples > 0 else 0
            avg_score = sum(scores) / len(scores) if scores else 0
            median_score = np.median(scores) if scores else 0
            std_score = np.std(scores) if scores else 0
            min_score = min(scores) if scores else 0
            max_score = max(scores) if scores else 0
            
            # Calculate percentiles
            percentiles = {}
            if scores:
                percentiles = {
                    'p25': np.percentile(scores, 25),
                    'p50': np.percentile(scores, 50),
                    'p75': np.percentile(scores, 75),
                    'p90': np.percentile(scores, 90),
                    'p95': np.percentile(scores, 95)
                }
            
            analysis[model_name] = {
                'total_samples': total_samples,
                'pass_count': pass_count,
                'pass_rate': pass_rate,
                'avg_score': avg_score,
                'median_score': median_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'percentiles': percentiles,
                'scores': scores,
                'score_ranges': dict(score_ranges),
                'error_counts': dict(error_counts),
                'sample_errors': sample_errors
            }
            
            print(f"📈 {model_name}:")
            print(f"  Total Samples: {total_samples}")
            print(f"  Pass Count: {pass_count} ({pass_rate:.2%})")
            print(f"  Score Statistics:")
            print(f"    Mean: {avg_score:.4f}")
            print(f"    Median: {median_score:.4f}")
            print(f"    Std Dev: {std_score:.4f}")
            print(f"    Range: [{min_score:.4f}, {max_score:.4f}]")
            if percentiles:
                print(f"    Percentiles: P25={percentiles['p25']:.3f}, P50={percentiles['p50']:.3f}, P75={percentiles['p75']:.3f}, P95={percentiles['p95']:.3f}")
            if error_counts:
                print(f"  Errors:")
                for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")
            print()
        
        self.analysis = analysis
        return analysis
    
    def create_pass_rate_chart(self):
        """Create pass rate comparison bar chart."""
        print(f"📊 Creating pass rate chart...")
        
        models = list(self.analysis.keys())
        pass_rates = [self.analysis[m]['pass_rate'] * 100 for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(models, pass_rates, color='steelblue')
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pass_rates[i]:.1f}%',
                   ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Pass Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Pass Rate Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 105)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "pass_rate_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_score_distribution_chart(self):
        """Create score distribution histogram for each model."""
        print(f"📊 Creating score distribution charts...")
        
        num_models = len(self.analysis)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        for ax, (model_name, data) in zip(axes, self.analysis.items()):
            scores = data['scores']
            
            ax.hist(scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Pass Threshold')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_name}\nScore Distribution', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "score_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_error_distribution_chart(self):
        """Create error distribution pie charts."""
        print(f"📊 Creating error distribution charts...")
        
        num_models = len(self.analysis)
        fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(range(8))
        
        for ax, (model_name, data) in zip(axes, self.analysis.items()):
            # Prepare data
            categories = ['Success'] + list(data['error_counts'].keys())
            values = [data['pass_count']] + list(data['error_counts'].values())
            
            if sum(values) == 0:
                continue
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=categories,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            # Style
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(f'{model_name}\nError Distribution', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "error_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_comparison_table(self):
        """Create detailed comparison table."""
        print(f"📊 Creating comparison table...")
        
        # Prepare data for table
        rows = []
        for model_name, data in self.analysis.items():
            rows.append({
                'Model': model_name,
                'Total': data['total_samples'],
                'Pass': data['pass_count'],
                'Pass Rate': f"{data['pass_rate']:.2%}",
                'Avg Score': f"{data['avg_score']:.4f}",
                'Errors': sum(data['error_counts'].values())
            })
        
        df = pd.DataFrame(rows)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(10, len(rows) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgray'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('steelblue')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        
        output_path = self.output_dir / "comparison_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_box_plot_comparison(self):
        """Create box plot comparison of score distributions."""
        print(f"📊 Creating box plot comparison...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        models = list(self.analysis.keys())
        score_data = [self.analysis[m]['scores'] for m in models]
        
        # Create box plot
        bp = ax.boxplot(score_data, labels=models, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Style boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        # Add reference line at 1.0 (pass threshold)
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Pass Threshold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Score Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "box_plot_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_percentile_comparison(self):
        """Create percentile comparison chart."""
        print(f"📊 Creating percentile comparison...")
        
        models = list(self.analysis.keys())
        percentiles = ['p25', 'p50', 'p75', 'p90', 'p95']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(percentiles))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [self.analysis[model]['percentiles'][p] for p in percentiles]
            ax.bar(x + i * width, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Score Percentile Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(['25th', '50th', '75th', '90th', '95th'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / "percentile_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_detailed_metrics_table(self):
        """Create detailed metrics table with statistics."""
        print(f"📊 Creating detailed metrics table...")
        
        rows = []
        for model_name, data in self.analysis.items():
            rows.append({
                'Model': model_name,
                'Samples': data['total_samples'],
                'Pass Rate': f"{data['pass_rate']:.2%}",
                'Mean': f"{data['avg_score']:.4f}",
                'Median': f"{data['median_score']:.4f}",
                'Std Dev': f"{data['std_score']:.4f}",
                'Min': f"{data['min_score']:.4f}",
                'Max': f"{data['max_score']:.4f}",
                'P95': f"{data['percentiles'].get('p95', 0):.4f}"
            })
        
        df = pd.DataFrame(rows)
        
        fig, ax = plt.subplots(figsize=(14, len(rows) * 0.6 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgray'] * len(df.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('steelblue')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Detailed Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        
        output_path = self.output_dir / "detailed_metrics_table.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def create_cumulative_distribution(self):
        """Create cumulative distribution function plot."""
        print(f"📊 Creating cumulative distribution plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, data in self.analysis.items():
            scores = sorted(data['scores'])
            cumulative = np.arange(1, len(scores) + 1) / len(scores)
            ax.plot(scores, cumulative, label=model_name, linewidth=2, marker='o', 
                   markersize=1, alpha=0.7)
        
        ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Pass Threshold')
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "cumulative_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def save_json_summary(self):
        """Save detailed analysis as JSON."""
        print(f"💾 Saving JSON summary...")
        
        summary = {
            'evaluation_id': self.eval_id,
            'evaluation_name': self.evaluation.name if self.evaluation else None,
            'analysis_date': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, data in self.analysis.items():
            summary['models'][model_name] = {
                'total_samples': data['total_samples'],
                'pass_count': data['pass_count'],
                'pass_rate': data['pass_rate'],
                'statistics': {
                    'mean': data['avg_score'],
                    'median': data['median_score'],
                    'std_dev': data['std_score'],
                    'min': data['min_score'],
                    'max': data['max_score']
                },
                'percentiles': data['percentiles'],
                'error_counts': data['error_counts']
            }
        
        output_path = self.output_dir / "analysis_summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Saved to {output_path}\n")
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("="*70)
        print("🎯 EVALUATION RUN ANALYSIS")
        print("="*70)
        print()
        
        # Fetch data
        self.fetch_evaluation()
        self.fetch_runs()
        self.fetch_run_results()
        
        # Analyze
        self.analyze_results()
        
        # Generate visualizations
        print("="*70)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*70)
        print()
        
        self.create_pass_rate_chart()
        self.create_score_distribution_chart()
        self.create_error_distribution_chart()
        self.create_box_plot_comparison()
        self.create_percentile_comparison()
        self.create_cumulative_distribution()
        self.create_comparison_table()
        self.create_detailed_metrics_table()
        self.save_json_summary()
        
        print("="*70)
        print(f"✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze Azure OpenAI evaluation runs',
        usage='%(prog)s RESOURCE_NAME EVAL_ID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/analyze_eval_run.py omi-fdp-swc-resource eval_691abff78ca88191ad5c8ffd039a40d1
        """
    )
    
    parser.add_argument(
        'resource_name',
        help='Azure OpenAI resource name'
    )
    
    parser.add_argument(
        'eval_id',
        help='Evaluation ID to analyze'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = EvaluationAnalyzer(
            eval_id=args.eval_id,
            resource_name=args.resource_name
        )
        analyzer.run_analysis()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
