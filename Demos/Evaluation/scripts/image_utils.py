"""
Image Utilities for Azure OpenAI Evaluation
Utilities for creating and processing image files for evaluation.
"""

from datasets import load_dataset
import json

MAX_SAMPLES = 100
OUTPUT_FILE = "./data/image_emotion_evaluation.jsonl"

def load_and_create_image_dataset(dataset_id: str, max_samples: int = MAX_SAMPLES):
    # Load the dataset (train split)
    dataset = load_dataset(dataset_id, split="train")

    eval_data = []

    for i in range(min(len(dataset), max_samples)):
        try:
            image_url = dataset[i]["image_url"]
            caption = dataset[i]["caption"]

            # Create evaluation item with image URL and caption
            eval_item = {
                "item": {
                    "image_url": image_url,
                    "caption": caption
                }
            }

            eval_data.append(eval_item)

        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
            continue

     # Write to JSONL file
    with open(OUTPUT_FILE, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Created evaluation file with {len(eval_data)} items")


def display_items(num_lines: int = 10):
    with open(OUTPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            item = json.loads(line)
            print(json.dumps(item, indent=2))