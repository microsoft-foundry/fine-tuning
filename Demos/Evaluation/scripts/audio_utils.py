"""
Audio Utilities for Azure OpenAI Evaluation
Utilities for creating and processing audio files for evaluation.
"""

from datasets import load_dataset, Audio
import base64
import json
import numpy as np
import io
import soundfile as sf
from pathlib import Path

MAX_SAMPLES = 100
OUTPUT_FILE = "./data/audio_emotion_evaluation.jsonl"

def load_and_create_audio_dataset(dataset_id: str, max_samples: int = MAX_SAMPLES):
    # Load only the requested slice and keep encoded WAV bytes to avoid FFmpeg.
    dataset = load_dataset(dataset_id, split=f"train[:{max_samples}]")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    eval_data = []

    for i, row in enumerate(dataset):
        try:
            item = row["audio"]
            emotion = row["major_emotion"]

            audio_base64 = _audio_to_base64(item)

            # Create evaluation item with real audio
            eval_item = {
                "item": {
                    "audio_data": audio_base64,
                    "expected_emotion": emotion
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


def _audio_to_base64(item) -> str:
    if item.get("bytes"):
        return base64.b64encode(item["bytes"]).decode("ascii")

    array = np.asarray(item["array"], dtype=np.float32)
    sr = int(item["sampling_rate"])

    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    wav_bytes = buf.read()
    return base64.b64encode(wav_bytes).decode("ascii")