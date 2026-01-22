"""
RFT Test Scenario Analysis Tool

Analyzes a specific test scenario across RFT training steps to show how responses improve.
Shows actual model outputs and scores for all variations of a scenario across steps.

Usage:
    python tools/analyze_rft_test_scenario.py
    python tools/analyze_rft_test_scenario.py --scenario "Your scenario text here"
    python tools/analyze_rft_test_scenario.py --scenario-id 0
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class RFTScenarioAnalyzer:
    """Analyze a specific scenario across RFT training steps."""
    
    def __init__(self, scenario_text=None, scenario_id=None):
        """Initialize analyzer with scenario identifier."""
        self.data_dir = Path("analysis_charts/rft_eval/data")
        self.output_dir = Path("analysis_charts/rft_eval/scenario_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scenario_text = scenario_text
        self.scenario_id = scenario_id
        self.steps_data = {}
        self.steps = []
        self.target_scenario = None
        self.scenario_data = {}
        
    def load_data(self):
        """Load all step data from JSON files."""
        print("="*70)
        print("RFT SCENARIO ANALYSIS")
        print("="*70)
        print()
        print(f"Loading data from {self.data_dir}...\n")
        
        # Find all step files
        step_files = sorted(self.data_dir.glob("Step_*.json"))
        
        if not step_files:
            raise FileNotFoundError(f"No step data files found in {self.data_dir}")
        
        for file in step_files:
            # Extract step number from filename
            step_num = int(file.stem.split("_")[1])
            
            with open(file, 'r') as f:
                data = json.load(f)
            
            print(f"  Loaded {file.name}: {data['total_items']} items")
            
            self.steps.append(step_num)
            self.steps_data[step_num] = data
        
        self.steps.sort()
        print(f"\n  Found {len(self.steps)} steps: {self.steps}\n")
    
    def identify_scenario(self):
        """Identify the target scenario from the data."""
        print("Identifying target scenario...\n")
        
        # Get all unique scenarios from Step 0
        step0_items = self.steps_data[self.steps[0]]['items']
        scenarios = []
        
        for item in step0_items:
            if 'datasource_item' in item and item['datasource_item']:
                user_msg = item['datasource_item'].get('reference_user_message', '')
                if user_msg and user_msg not in [s['text'] for s in scenarios]:
                    scenarios.append({
                        'id': len(scenarios),
                        'text': user_msg,
                        'preview': user_msg[:80] + '...' if len(user_msg) > 80 else user_msg
                    })
        
        print(f"  Found {len(scenarios)} unique scenarios\n")
        
        # Select scenario based on input
        if self.scenario_id is not None:
            if 0 <= self.scenario_id < len(scenarios):
                self.target_scenario = scenarios[self.scenario_id]
                print(f"  Using Scenario ID {self.scenario_id}")
            else:
                raise ValueError(f"Scenario ID {self.scenario_id} out of range (0-{len(scenarios)-1})")
        
        elif self.scenario_text:
            # Find matching scenario
            for scenario in scenarios:
                if self.scenario_text in scenario['text']:
                    self.target_scenario = scenario
                    print(f"  Found matching scenario (ID {scenario['id']})")
                    break
            
            if not self.target_scenario:
                print("  WARNING: No exact match found. Available scenarios:")
                for s in scenarios:
                    print(f"    [{s['id']}] {s['preview']}")
                raise ValueError(f"Scenario not found: {self.scenario_text}")
        
        else:
            # Use default (first scenario)
            self.target_scenario = scenarios[0]
            print(f"  Using default scenario (ID 0)")
        
        print(f"\n  Selected Scenario:")
        print(f"     ID: {self.target_scenario['id']}")
        print(f"     Text: {self.target_scenario['text'][:150]}")
        if len(self.target_scenario['text']) > 150:
            print(f"           {self.target_scenario['text'][150:300]}")
        print()
    
    def extract_scenario_data(self):
        """Extract all data for the target scenario across steps."""
        print("Extracting scenario data across steps...\n")
        
        target_text = self.target_scenario['text']
        
        for step in self.steps:
            items = self.steps_data[step]['items']
            variations = []
            
            for item in items:
                if 'datasource_item' in item and item['datasource_item']:
                    user_msg = item['datasource_item'].get('reference_user_message', '')
                    
                    if user_msg == target_text:
                        # Extract relevant data
                        score = 0.0
                        if 'results' in item and item['results']:
                            score = item['results'][0].get('score', 0.0)
                        
                        # Get model output
                        output = None
                        if 'sample' in item and item['sample']:
                            output = item['sample'].get('output', {})
                        
                        # Get reference data
                        ref_tool_calls = item['datasource_item'].get('reference_tool_calls', [])
                        ref_policy_args = item['datasource_item'].get('reference_policy_args', [])
                        
                        variations.append({
                            'score': score,
                            'output': output,
                            'reference_tool_calls': ref_tool_calls,
                            'reference_policy_args': ref_policy_args,
                            'item': item
                        })
            
            self.scenario_data[step] = variations
            print(f"  Step {step:2d}: Found {len(variations)} variations")
        
        print()
    
    def extract_tool_calls(self, output):
        """Extract tool call names from output."""
        if not output or not isinstance(output, dict):
            return []
        
        tool_calls = output.get('tool_calls', [])
        if not tool_calls:
            return []
        
        return [tc.get('function', {}).get('name', 'unknown') for tc in tool_calls]
    
    def format_json_output(self, output):
        """Format output as nicely formatted JSON string."""
        if not output:
            return "No response"
        
        if isinstance(output, dict):
            # Try to format as JSON
            try:
                return json.dumps(output, indent=2)
            except:
                return str(output)
        else:
            return str(output)
    
    def create_summary_table(self):
        """Create a summary table showing progression across steps."""
        print("Creating summary table...\n")
        
        # Prepare data for table
        rows = []
        
        for step in self.steps:
            variations = self.scenario_data[step]
            
            for i, var in enumerate(variations, 1):
                # Extract tool calls
                tool_calls = self.extract_tool_calls(var['output'])
                tool_calls_str = ', '.join(tool_calls) if tool_calls else 'None'
                
                # Get message content if available
                content = None
                if var['output'] and isinstance(var['output'], dict):
                    content = var['output'].get('content', '')
                
                rows.append({
                    'Step': step,
                    'Variation': i,
                    'Score': f"{var['score']:.2f}",
                    'Tool Calls': tool_calls_str[:50] + ('...' if len(tool_calls_str) > 50 else ''),
                    'Response Preview': content[:80] + '...' if content and len(content) > 80 else (content or 'N/A')
                })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_file = self.output_dir / f"scenario_{self.target_scenario['id']}_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"  Saved CSV: {csv_file}")
        
        # Save detailed JSON
        json_file = self.output_dir / f"scenario_{self.target_scenario['id']}_detailed.json"
        with open(json_file, 'w') as f:
            json.dump({
                'scenario': self.target_scenario,
                'steps_data': {str(k): v for k, v in self.scenario_data.items()}
            }, f, indent=2)
        print(f"  Saved JSON: {json_file}")
        
        return df
    
    def create_score_comparison_chart(self, df):
        """Create a chart showing score progression for each variation."""
        print("Creating score comparison chart...\n")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Chart 1: Line plot for each variation
        for variation_id in sorted(df['Variation'].unique()):
            var_data = df[df['Variation'] == variation_id]
            steps = var_data['Step'].values
            scores = var_data['Score'].astype(float).values
            
            ax1.plot(steps, scores, marker='o', linewidth=2, markersize=8,
                    label=f'Variation {variation_id}', alpha=0.7)
        
        ax1.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Score Progression by Variation', fontsize=14, fontweight='bold')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='best')
        
        # Chart 2: Box plot showing score distribution per step
        step_data = [df[df['Step'] == step]['Score'].astype(float).values 
                     for step in sorted(df['Step'].unique())]
        
        bp = ax2.boxplot(step_data, labels=[str(s) for s in sorted(df['Step'].unique())],
                        patch_artist=True, showmeans=True)
        
        # Color the boxes
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(step_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('RFT Training Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Score Distribution by Step', fontsize=14, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Scenario {self.target_scenario["id"]}: Performance Across RFT Steps', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_file = self.output_dir / f"scenario_{self.target_scenario['id']}_scores.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved chart: {output_file}")
    
    def create_detailed_comparison_table(self):
        """Create a detailed table comparing responses across steps."""
        print("Creating detailed comparison table...\n")
        
        # Create figure with space for table and examples table
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid: top for scores table, bottom for examples table
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
        
        # ==================== SCORES TABLE ====================
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        # Prepare table data
        variations_count = len(self.scenario_data[self.steps[0]])
        
        # Header
        headers = ['Var'] + [f'Step {s}' for s in self.steps]
        
        # Rows - one per variation (simplified - just scores)
        table_data = []
        for var_idx in range(variations_count):
            row = [f'V{var_idx + 1}']
            
            for step in self.steps:
                variations = self.scenario_data[step]
                if var_idx < len(variations):
                    var = variations[var_idx]
                    score = var['score']
                    row.append(f"{score:.2f}")
                else:
                    row.append('N/A')
            
            table_data.append(row)
        
        # Create table
        table = ax_table.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='center',
                              colWidths=[0.08] + [0.92 / len(self.steps)] * len(self.steps))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color cells based on score
        for i in range(1, len(table_data) + 1):
            for j in range(1, len(headers)):
                step = self.steps[j - 1]
                var_idx = i - 1
                
                if var_idx < len(self.scenario_data[step]):
                    score = self.scenario_data[step][var_idx]['score']
                    
                    # Color based on score
                    if score >= 0.8:
                        color = '#90EE90'  # Light green
                    elif score >= 0.5:
                        color = '#FFD700'  # Gold
                    else:
                        color = '#FFB6C1'  # Light red
                    
                    table[(i, j)].set_facecolor(color)
                    table[(i, j)].set_alpha(0.6)
        
        # Style header
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)
        
        # Add legend for scores
        legend_text = (
            "Score Legend:\n"
            "  0.0 = Got eligibility and policy reason wrong\n"
            "  0.5 = Got eligibility correct but no policy reason\n"
            "  1.0 = Got both eligibility and policy reason correct"
        )
        ax_table.text(0.02, 0.98, legend_text, transform=ax_table.transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax_table.set_title(f'Score Progression - Scenario {self.target_scenario["id"]}\n'
                          f'{self.target_scenario["preview"]}',
                          fontsize=13, fontweight='bold', pad=15)
        
        # ==================== EXAMPLES TABLE ====================
        # Find worst example from Step 0 and best from final step
        step0_variations = self.scenario_data[self.steps[0]]
        step_final_variations = self.scenario_data[self.steps[-1]]
        
        # Find lowest score in Step 0
        worst_var = min(step0_variations, key=lambda x: x['score'])
        worst_score = worst_var['score']
        worst_output_formatted = self.format_json_output(worst_var['output'])
        
        # Find highest score in final step
        best_var = max(step_final_variations, key=lambda x: x['score'])
        best_score = best_var['score']
        best_output_formatted = self.format_json_output(best_var['output'])
        
        # Truncate if needed and escape special characters for matplotlib
        max_chars = 800
        if len(worst_output_formatted) > max_chars:
            worst_output_formatted = worst_output_formatted[:max_chars] + "\n... (truncated)"
        if len(best_output_formatted) > max_chars:
            best_output_formatted = best_output_formatted[:max_chars] + "\n... (truncated)"
        
        # Escape special characters to prevent matplotlib from interpreting them as LaTeX
        def escape_for_matplotlib(text):
            # Escape backslashes, dollar signs, and other special chars
            text = text.replace('\\', '\\\\')
            text = text.replace('$', r'\$')
            text = text.replace('_', r'\_')
            text = text.replace('^', r'\^')
            text = text.replace('{', r'\{')
            text = text.replace('}', r'\}')
            return text
        
        worst_output_formatted = escape_for_matplotlib(worst_output_formatted)
        best_output_formatted = escape_for_matplotlib(best_output_formatted)
        
        # Create unified examples table
        ax_examples = fig.add_subplot(gs[1])
        ax_examples.axis('off')
        
        example_data = [
            [f'Step {self.steps[0]}\n(Worst)', f'{worst_score:.2f}', worst_output_formatted],
            [f'Step {self.steps[-1]}\n(Best)', f'{best_score:.2f}', best_output_formatted]
        ]
        
        examples_table = ax_examples.table(
            cellText=example_data,
            colLabels=['Step', 'Score', 'Model Output (JSON)'],
            colWidths=[0.10, 0.08, 0.82],
            cellLoc='left',
            loc='center'
        )
        
        # Style the table
        examples_table.auto_set_font_size(False)
        examples_table.scale(1, 3)
        
        # Style header row
        for i in range(3):
            cell = examples_table[(0, i)]
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(weight='bold', size=9)
        
        # Style data rows with color coding
        for row in range(1, 3):
            # Step column
            examples_table[(row, 0)].set_text_props(size=8, weight='bold')
            # Score column
            examples_table[(row, 1)].set_text_props(size=8, weight='bold')
            # Output column - monospace font
            examples_table[(row, 2)].set_text_props(size=8, family='monospace')
            
            # Color code: red for worst (row 1), green for best (row 2)
            if row == 1:
                examples_table[(row, 0)].set_facecolor('#FFB6C1')  # Light red
                examples_table[(row, 1)].set_facecolor('#FFB6C1')
                examples_table[(row, 2)].set_facecolor('#FFE4E1')  # Very light red
            else:
                examples_table[(row, 0)].set_facecolor('#90EE90')  # Light green
                examples_table[(row, 1)].set_facecolor('#90EE90')
                examples_table[(row, 2)].set_facecolor('#E0FFE0')  # Very light green
        
        ax_examples.set_title('Example Comparison: Worst vs Best Response',
                             fontsize=11, fontweight='bold', pad=10)
        
        output_file = self.output_dir / f"scenario_{self.target_scenario['id']}_detailed_table.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved detailed table: {output_file}")
    
    def create_full_output_comparison(self):
        """Save full model outputs to text files for easy comparison."""
        print("Creating full output comparison...\n")
        
        # Find worst example from Step 0 and best from final step
        step0_variations = self.scenario_data[self.steps[0]]
        step_final_variations = self.scenario_data[self.steps[-1]]
        
        worst_var = min(step0_variations, key=lambda x: x['score'])
        best_var = max(step_final_variations, key=lambda x: x['score'])
        
        # Extract and decode full content
        def get_full_content(var):
            output = var['output']
            if not output:
                return "No response"
            
            # Handle list output (multiple messages)
            if isinstance(output, list):
                contents = []
                for item in output:
                    if isinstance(item, dict) and 'content' in item:
                        contents.append(item['content'])
                return '\n\n'.join(contents) if contents else json.dumps(output, indent=2, ensure_ascii=False)
            
            # Handle dict output
            elif isinstance(output, dict):
                if 'content' in output:
                    return output['content']
                return json.dumps(output, indent=2, ensure_ascii=False)
            
            else:
                return str(output)
        
        worst_content = get_full_content(worst_var)
        best_content = get_full_content(best_var)
        
        # Save worst response to file
        worst_file = self.output_dir / f"scenario_{self.target_scenario['id']}_worst_output.txt"
        with open(worst_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"WORST RESPONSE - Scenario {self.target_scenario['id']}\n")
            f.write(f"Step: {self.steps[0]}\n")
            f.write(f"Score: {worst_var['score']:.2f}\n")
            f.write(f"Scenario: {self.target_scenario['preview']}\n")
            f.write("="*80 + "\n\n")
            f.write(worst_content)
            f.write("\n\n" + "="*80 + "\n")
        
        # Save best response to file
        best_file = self.output_dir / f"scenario_{self.target_scenario['id']}_best_output.txt"
        with open(best_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BEST RESPONSE - Scenario {self.target_scenario['id']}\n")
            f.write(f"Step: {self.steps[-1]}\n")
            f.write(f"Score: {best_var['score']:.2f}\n")
            f.write(f"Scenario: {self.target_scenario['preview']}\n")
            f.write("="*80 + "\n\n")
            f.write(best_content)
            f.write("\n\n" + "="*80 + "\n")
        
        # Save side-by-side comparison
        comparison_file = self.output_dir / f"scenario_{self.target_scenario['id']}_comparison.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write(f"OUTPUT COMPARISON - Scenario {self.target_scenario['id']}\n")
            f.write(f"Scenario: {self.target_scenario['preview']}\n")
            f.write("="*120 + "\n\n")
            
            f.write(f"WORST RESPONSE (Step {self.steps[0]}, Score: {worst_var['score']:.2f})\n")
            f.write("-"*120 + "\n")
            f.write(worst_content)
            f.write("\n\n" + "="*120 + "\n\n")
            
            f.write(f"BEST RESPONSE (Step {self.steps[-1]}, Score: {best_var['score']:.2f})\n")
            f.write("-"*120 + "\n")
            f.write(best_content)
            f.write("\n\n" + "="*120 + "\n")
        
        print(f"  Saved worst output: {worst_file}")
        print(f"  Saved best output: {best_file}")
        print(f"  Saved comparison: {comparison_file}")
    
    def print_detailed_outputs(self):
        """Print detailed outputs for manual inspection."""
        print("\n" + "="*70)
        print("DETAILED OUTPUTS")
        print("="*70)
        
        for step in self.steps:
            print(f"\n{'='*70}")
            print(f"STEP {step}")
            print('='*70)
            
            variations = self.scenario_data[step]
            
            for i, var in enumerate(variations, 1):
                print(f"\n--- Variation {i} ---")
                print(f"Score: {var['score']:.2f}")
                
                tool_calls = self.extract_tool_calls(var['output'])
                print(f"Tool Calls: {', '.join(tool_calls) if tool_calls else 'None'}")
                
                if var['output'] and isinstance(var['output'], dict):
                    content = var['output'].get('content', '')
                    if content:
                        print(f"Response: {content[:200]}")
                        if len(content) > 200:
                            print(f"          ... (truncated)")
    
    def run_analysis(self):
        """Run complete scenario analysis pipeline."""
        self.load_data()
        self.identify_scenario()
        self.extract_scenario_data()
        
        # Create outputs
        df = self.create_summary_table()
        self.create_score_comparison_chart(df)
        self.create_detailed_comparison_table()
        self.create_full_output_comparison()
        self.print_detailed_outputs()
        
        print("\n" + "="*70)
        print("SCENARIO ANALYSIS COMPLETE")
        print("="*70)
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Scenario ID: {self.target_scenario['id']}")
        print(f"Steps analyzed: {len(self.steps)}")
        print(f"Variations per step: {len(self.scenario_data[self.steps[0]])}")
        
        # Calculate improvement
        first_step_scores = [v['score'] for v in self.scenario_data[self.steps[0]]]
        last_step_scores = [v['score'] for v in self.scenario_data[self.steps[-1]]]
        
        avg_first = sum(first_step_scores) / len(first_step_scores)
        avg_last = sum(last_step_scores) / len(last_step_scores)
        improvement = ((avg_last - avg_first) / avg_first) * 100 if avg_first > 0 else 0
        
        print(f"\nPerformance Summary:")
        print(f"   Step {self.steps[0]} avg: {avg_first:.3f}")
        print(f"   Step {self.steps[-1]} avg: {avg_last:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze a specific scenario across RFT training steps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default scenario
    python tools/analyze_rft_test_scenario.py
    
    # Use specific scenario ID
    python tools/analyze_rft_test_scenario.py --scenario-id 0
    
    # Search by scenario text
    python tools/analyze_rft_test_scenario.py --scenario "electric kettle"
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        help='Scenario text to search for (partial match allowed)'
    )
    
    parser.add_argument(
        '--scenario-id',
        type=int,
        help='Scenario ID (0-based index)'
    )
    
    args = parser.parse_args()
    
    # Default scenario if none provided
    if not args.scenario and args.scenario_id is None:
        args.scenario_id = 9  # Default: "The electric kettle I bought is leaking. I'm Noah Brown, ZIP 80279."
    
    try:
        analyzer = RFTScenarioAnalyzer(
            scenario_text=args.scenario,
            scenario_id=args.scenario_id
        )
        analyzer.run_analysis()
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
