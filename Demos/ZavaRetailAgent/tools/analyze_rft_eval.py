"""
RFT Evaluation Analysis Tool

Downloads and analyzes Reinforced Fine-Tuning (RFT) evaluation runs across multiple steps.
This script tracks model improvement across RFT training steps (Step 0, 3, 6, 9).

Usage:
    python tools/analyze_rft_eval.py RESOURCE_NAME EVAL_ID
    python tools/analyze_rft_eval.py omi-ignite-demo-resource eval_691c31f82bf0819199e55210bf0595a0
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


class RFTEvaluationDownloader:
    """Download RFT evaluation data across training steps."""
    
    def __init__(self, eval_id, resource_name):
        """Initialize downloader with evaluation ID."""
        self.eval_id = eval_id
        self.resource_name = resource_name
        self.output_dir = Path("analysis_charts/rft_eval")
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.run_data = {}
    
    def to_serializable(self, obj):
        """Convert Pydantic models and other objects to JSON-serializable format."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: self.to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(item) for item in obj]
        else:
            return obj
        self.runs = []
        self.run_data = {}
    
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
            
            # Sort runs by step number
            step_runs = []
            for run in self.runs:
                print(f"  - {run.name} (ID: {run.id})")
                if hasattr(run, 'status'):
                    print(f"    Status: {run.status}")
                
                # Extract step number from name
                step = None
                if "Step" in run.name:
                    try:
                        step = int(run.name.split("Step")[1].strip().split()[0])
                    except:
                        pass
                step_runs.append((step if step is not None else -1, run))
            
            # Sort by step
            step_runs.sort(key=lambda x: x[0])
            self.runs = [run for _, run in step_runs]
            
            print()
        except Exception as e:
            print(f"❌ Error fetching runs: {e}")
            raise
    
    def download_run_data(self, run):
        """Download all output items for a single run."""
        print(f"  📥 Downloading data for {run.name}...")
        try:
            output_items = self.client.evals.runs.output_items.list(
                eval_id=self.eval_id,
                run_id=run.id
            )
            
            items = list(output_items)
            print(f"    ✓ Retrieved {len(items)} items")
            
            # Convert items to serializable format
            items_data = []
            for item in items:
                item_dict = {
                    'id': item.id if hasattr(item, 'id') else None,
                    'status': item.status if hasattr(item, 'status') else None,
                    'results': []
                }
                
                # Extract results
                if hasattr(item, 'results') and item.results:
                    for result in item.results:
                        result_dict = {}
                        if hasattr(result, 'score'):
                            result_dict['score'] = result.score
                        if hasattr(result, 'name'):
                            result_dict['name'] = result.name
                        if hasattr(result, 'pass_'):
                            result_dict['pass'] = result.pass_
                        if hasattr(result, 'trace'):
                            result_dict['trace'] = result.trace
                        if hasattr(result, 'output'):
                            result_dict['output'] = result.output
                        item_dict['results'].append(result_dict)
                
                # Extract sample data (input/output from model)
                if hasattr(item, 'sample') and item.sample:
                    sample = item.sample
                    sample_dict = {}
                    if hasattr(sample, 'input'):
                        input_val = sample.input
                        # Convert Pydantic models to dict
                        if hasattr(input_val, 'model_dump'):
                            sample_dict['input'] = input_val.model_dump()
                        elif hasattr(input_val, 'dict'):
                            sample_dict['input'] = input_val.dict()
                        else:
                            sample_dict['input'] = input_val
                    
                    if hasattr(sample, 'output'):
                        output_val = sample.output
                        # Convert Pydantic models to dict
                        if hasattr(output_val, 'model_dump'):
                            sample_dict['output'] = output_val.model_dump()
                        elif hasattr(output_val, 'dict'):
                            sample_dict['output'] = output_val.dict()
                        else:
                            sample_dict['output'] = output_val
                    
                    item_dict['sample'] = sample_dict
                
                # Extract datasource item (reference data and metadata)
                if hasattr(item, 'datasource_item') and item.datasource_item:
                    ds = item.datasource_item
                    if isinstance(ds, dict):
                        item_dict['datasource_item'] = {
                            'messages': self.to_serializable(ds.get('messages')),
                            'tools': self.to_serializable(ds.get('tools')),
                            'parallel_tool_calls': ds.get('parallel_tool_calls'),
                            'reference_tool_calls': ds.get('reference_tool_calls'),
                            'reference_policy_args': self.to_serializable(ds.get('reference_policy_args')),
                            'reference_policy_outcome': self.to_serializable(ds.get('reference_policy_outcome')),
                            'reference_developer_message': ds.get('reference_developer_message'),
                            'reference_user_message': ds.get('reference_user_message'),
                            'trace': ds.get('trace')
                        }
                
                items_data.append(item_dict)
            
            return {
                'run_id': run.id,
                'run_name': run.name,
                'status': run.status if hasattr(run, 'status') else 'unknown',
                'total_items': len(items),
                'items': items_data,
                'downloaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return {
                'run_id': run.id,
                'run_name': run.name,
                'error': str(e),
                'total_items': 0,
                'items': []
            }
    
    def download_all_runs(self):
        """Download data for all runs."""
        print(f"📦 Downloading data for all runs...\n")
        
        for run in self.runs:
            run_data = self.download_run_data(run)
            self.run_data[run.name] = run_data
            
            # Save individual run data (serialize it first)
            output_file = self.data_dir / f"{run.name.replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                serializable_data = self.to_serializable(run_data)
                json.dump(serializable_data, f, indent=2)
            print(f"    💾 Saved to {output_file}")
            print()
    
    def save_combined_data(self):
        """Save all run data in a single file."""
        print(f"💾 Saving combined data...")
        
        combined_data = {
            'evaluation_id': self.eval_id,
            'evaluation_name': self.evaluation.name if self.evaluation else None,
            'resource_name': self.resource_name,
            'download_date': datetime.now().isoformat(),
            'total_runs': len(self.runs),
            'runs': self.run_data
        }
        
        output_file = self.data_dir / "combined_data.json"
        with open(output_file, 'w') as f:
            serializable_data = self.to_serializable(combined_data)
            json.dump(serializable_data, f, indent=2)
        
        print(f"  ✓ Saved to {output_file}")
        
        # Also save a summary
        summary = {
            'evaluation_id': self.eval_id,
            'evaluation_name': self.evaluation.name if self.evaluation else None,
            'download_date': datetime.now().isoformat(),
            'runs_summary': []
        }
        
        for run_name, data in self.run_data.items():
            summary['runs_summary'].append({
                'name': run_name,
                'run_id': data.get('run_id'),
                'status': data.get('status'),
                'total_items': data.get('total_items'),
                'has_error': 'error' in data
            })
        
        summary_file = self.data_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Summary saved to {summary_file}\n")
    
    def run_download(self):
        """Run complete download pipeline."""
        print("="*70)
        print("🎯 RFT EVALUATION DATA DOWNLOAD")
        print("="*70)
        print()
        
        # Fetch metadata
        self.fetch_evaluation()
        self.fetch_runs()
        
        # Download all data
        self.download_all_runs()
        
        # Save combined data
        self.save_combined_data()
        
        print("="*70)
        print(f"✅ DOWNLOAD COMPLETE")
        print("="*70)
        print(f"📁 Data directory: {self.data_dir.absolute()}")
        print(f"📊 Total runs downloaded: {len(self.runs)}")
        print()
        print("Next steps:")
        print("  1. Review the downloaded data in JSON files")
        print("  2. Run analysis on the data to generate charts")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download RFT evaluation data',
        usage='%(prog)s RESOURCE_NAME EVAL_ID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/analyze_rft_eval.py omi-ignite-demo-resource eval_691c31f82bf0819199e55210bf0595a0
        """
    )
    
    parser.add_argument(
        'resource_name',
        help='Azure OpenAI resource name'
    )
    
    parser.add_argument(
        'eval_id',
        help='Evaluation ID to download'
    )
    
    args = parser.parse_args()
    
    try:
        downloader = RFTEvaluationDownloader(
            eval_id=args.eval_id,
            resource_name=args.resource_name
        )
        downloader.run_download()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
