# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "pillow",
#     "tqdm",
# ]
# ///

"""Dataset loader for ESP32-CAM robot data"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import json


class ESP32RobotDataset(Dataset):
    """
    Dataset for ESP32-CAM robot with IMU sensors
    
    Expected data structure:
    data/robot_dataset/
        ├── episode_0000/
        │   ├── images/
        │   │   ├── 0000.jpg
        │   │   └── ...
        │   ├── sensor_data.json
        │   └── actions.json
        └── episode_0001/
            └── ...
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: Tuple[int, int] = (224, 224),
        augmentation: bool = False,
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.augmentation = augmentation
        
        # Load all episodes
        self.episodes = []
        self.episode_lengths = []
        self._load_episodes()
        
        print(f"Loaded {len(self.episodes)} episodes")
        print(f"Total frames: {sum(self.episode_lengths)}")
    
    def _load_episodes(self):
        """Load all episodes from disk"""
        episode_dirs = sorted(self.data_path.glob("episode_*"))
        
        for ep_dir in episode_dirs:
            try:
                episode_data = self._load_episode(ep_dir)
                self.episodes.append(episode_data)
                self.episode_lengths.append(len(episode_data['images']))
            except Exception as e:
                print(f"Warning: Failed to load {ep_dir}: {e}")
    
    def _load_episode(self, episode_dir: Path) -> Dict:
        """Load single episode data"""
        # Load sensor data (IMU)
        with open(episode_dir / "sensor_data.json", 'r') as f:
            sensor_data = json.load(f)
        
        # Load actions (motor commands)
        with open(episode_dir / "actions.json", 'r') as f:
            actions_data = json.load(f)
        
        # Get image paths
        image_files = sorted((episode_dir / "images").glob("*.jpg"))
        
        # Extract IMU states
        imu_states = []
        for frame in sensor_data['frames']:
            imu = frame['imu']
            state = [
                imu['accel']['x'], imu['accel']['y'], imu['accel']['z'],
                imu['gyro']['x'], imu['gyro']['y'], imu['gyro']['z']
            ]
            imu_states.append(state)
        
        # Extract motor actions
        actions = []
        for action in actions_data['frames']:
            actions.append([
                action['left_motor'],
                action['right_motor']
            ])
        
        return {
            'images': image_files,
            'imu_states': np.array(imu_states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'instruction': sensor_data.get('instruction', 'Move forward'),
        }
    
    def __len__(self) -> int:
        return sum(self.episode_lengths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find episode and frame index
        episode_idx, frame_idx = self._get_episode_frame(idx)
        episode = self.episodes[episode_idx]
        
        # Load and process image
        image = Image.open(episode['images'][frame_idx]).convert('RGB')
        image = image.resize(self.image_size)
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Apply augmentation if enabled
        if self.augmentation:
            image = self._augment_image(image)
        
        return {
            'image': torch.from_numpy(image).permute(2, 0, 1),  # HWC -> CHW
            'imu_state': torch.from_numpy(episode['imu_states'][frame_idx]),
            'action': torch.from_numpy(episode['actions'][frame_idx]),
            'instruction': episode['instruction'],
        }
    
    def _get_episode_frame(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (episode_idx, frame_idx)"""
        cumsum = 0
        for ep_idx, ep_len in enumerate(self.episode_lengths):
            if idx < cumsum + ep_len:
                return ep_idx, idx - cumsum
            cumsum += ep_len
        raise IndexError(f"Index {idx} out of range")
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Simple image augmentation"""
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)
        
        # Random horizontal flip (for robot navigation)
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
        
        return image
