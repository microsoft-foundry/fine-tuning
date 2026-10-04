"""
Image Utilities for Azure OpenAI Evaluation
Utilities for creating and processing image files for evaluation.
"""

from datasets import load_dataset
import json
from pathlib import Path

MAX_SAMPLES = 100
OUTPUT_FILE = "./data/image_emotion_evaluation.jsonl"

STABLE_IMAGE_SAMPLES = [
    {
        "image_url": "https://fastly.picsum.photos/id/10/512/512.jpg?hmac=Xno3CU-pY4UWutaA1XRUThqgbV_vn7kN5PjrlkRw-9Q",
        "caption": "A view of a calm ocean with distant islands, framed by lush green trees.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/20/512/512.jpg?hmac=pGfehaSsk_8XvaGx8DJNw0BHdYqBj6CZWPFu6sTNPiA",
        "caption": "A workspace featuring a laptop, smartphone, books, and design materials.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/30/512/512.jpg?hmac=ph0RnBNUKGfVgtH1-jDWPbN0TZXOXat6xKzPazurl_Y",
        "caption": "A white mug featuring a design inspired by a Cuban stamp.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/40/512/512.jpg?hmac=PIXIOTze0IyV8x16iptEpCgTW8UtUw1URQ7dp4YpXWk",
        "caption": "Close-up of a rabbit's nose and whiskers.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/50/512/512.jpg?hmac=f1wtpHWcDkYA2Mt_lilGmz2dWFPyedhIYGSf9t4ytKs",
        "caption": "A bird spreads its wings on a rope near a boat by the water.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/60/512/512.jpg?hmac=CKdYDj-ZhSmY_J11yU1Ep4CuKgy7LaPKpPY9MQkdzZA",
        "caption": "A top-down view of a workspace with a computer and stationery.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/70/512/512.jpg?hmac=cn7BM5mWG62CWbtJtqpvyNSPdF9YA1MbWv5ILxxnKSE",
        "caption": "A foggy road lined with tall trees.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/80/512/512.jpg?hmac=SPHV9-w09wNIQzeF8z166LbZFAPXcBnuNFYjguovcBU",
        "caption": "A black-and-white close-up of pine cones on wood.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/90/512/512.jpg?hmac=j6YDrXPg9rh47EZgUMw_AX_YfJs92PlQ7BbfJ_h7-4Y",
        "caption": "A row of glass jars placed on wooden stakes in a garden.",
    },
    {
        "image_url": "https://fastly.picsum.photos/id/100/512/512.jpg?hmac=sln0fKadK1l8jcObPqMkNXxeAjn9ZQu7taEEIx9k4c8",
        "caption": "A hazy beach scene with people near the water.",
    },
]


def load_and_create_image_dataset(dataset_id: str, max_samples: int = MAX_SAMPLES):
    if dataset_id == "picsum":
        dataset = STABLE_IMAGE_SAMPLES[:max_samples]
    else:
        dataset = load_dataset(dataset_id, split="train", streaming=True)

    eval_data = []

    for i, row in enumerate(dataset):
        if len(eval_data) >= max_samples:
            break

        try:
            image_url = row["image_url"]
            caption = row["caption"]

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

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

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