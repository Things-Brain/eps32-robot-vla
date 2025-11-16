#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24.0",
#     "pillow>=10.0.0",
#     "tqdm>=4.65.0",
# ]
# ///

"""
Prepare ESP32-CAM robot dataset from MQTT collected data
Usage: uv run scripts/prepare_data.py --input ./raw_data --output ./data/robot_dataset
"""

import json
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
from tqdm import tqdm


def process_episode(
    input_dir: Path,
    output_dir: Path,
    episode_id: int,
    resize: tuple = (224, 224)
) -> Dict:
    """Process single episode from MQTT data"""
    
    output_episode = output_dir / f"episode_{episode_id:04d}"
    output_episode.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory
    (output_episode / "images").mkdir(exist_ok=True)
    
    # Load raw sensor data
    with open(input_dir / "sensor_data.json", 'r') as f:
        sensor_data = json.load(f)
    
    # Load raw action data
    with open(input_dir / "actions.json", 'r') as f:
        actions_data = json.load(f)
    
    # Process and copy images
    image_files = sorted((input_dir / "images").glob("*.jpg"))
    for idx, img_path in enumerate(image_files):
        img = Image.open(img_path)
        img = img.resize(resize)
        img.save(output_episode / "images" / f"{idx:04d}.jpg", quality=95)
    
    # Save processed sensor data
    with open(output_episode / "sensor_data.json", 'w') as f:
        json.dump(sensor_data, f, indent=2)
    
    # Save processed actions
    with open(output_episode / "actions.json", 'w') as f:
        json.dump(actions_data, f, indent=2)
    
    return {
        'num_frames': len(image_files),
        'instruction': sensor_data.get('instruction', 'Move forward')
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ESP32 robot dataset for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with raw episode data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/robot_dataset",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (width height)"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all episodes
    episode_dirs = sorted(input_path.glob("episode_*"))
    
    if not episode_dirs:
        print(f"No episodes found in {input_path}")
        return
    
    print(f"Found {len(episode_dirs)} episodes")
    print(f"Processing to {output_path}")
    
    # Process all episodes
    episode_stats = []
    for idx, ep_dir in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
        try:
            stats = process_episode(
                ep_dir,
                output_path,
                idx,
                tuple(args.resize)
            )
            episode_stats.append(stats)
        except Exception as e:
            print(f"\nWarning: Failed to process {ep_dir}: {e}")
    
    # Create dataset metadata
    metadata = {
        'num_episodes': len(episode_stats),
        'total_frames': sum(s['num_frames'] for s in episode_stats),
        'action_dim': 2,
        'state_dim': 6,
        'image_size': args.resize,
        'description': 'ESP32-CAM robot dataset with IMU and motor control',
        'episodes': episode_stats,
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"Episodes: {metadata['num_episodes']}")
    print(f"Total frames: {metadata['total_frames']}")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
