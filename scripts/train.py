#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.35.0",
#     "peft>=0.7.0",
#     "accelerate>=0.25.0",
#     "wandb>=0.16.0",
#     "lerobot>=2.0.0",
#     "pyyaml>=6.0.0",
# ]
# ///

"""
SmolVLA Fine-tuning Training Script for ESP32-CAM Robot
Usage: uv run scripts/train.py --config config/smolvla_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import wandb
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from esp32_robot_vla.config import ESP32RobotConfig
from esp32_robot_vla.dataset import ESP32RobotDataset


def setup_wandb(config: ESP32RobotConfig):
    """Initialize Weights & Biases logging"""
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.run_name,
            config={
                "model": config.model.__dict__,
                "lora": config.lora.__dict__,
                "training": config.training.__dict__,
            }
        )


def create_lora_config(config: ESP32RobotConfig) -> LoraConfig:
    """Create LoRA configuration"""
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )


def train_lerobot(config: ESP32RobotConfig):
    """
    Train using LeRobot framework (recommended approach)
    """
    print("=" * 60)
    print("SmolVLA Fine-tuning with LeRobot")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.data.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {config.training.device}")
    print("=" * 60)
    
    # Build LeRobot training command
    cmd = [
        "python", "-m", "lerobot.scripts.train",
        f"--policy.path={config.model.name}",
        f"--dataset.repo_id={config.data.dataset_path}",
        f"--batch_size={config.training.batch_size}",
        f"--steps={config.training.num_train_steps}",
        f"--output_dir={config.output_dir}",
        f"--job_name={config.wandb.run_name}",
        f"--policy.device={config.training.device}",
        f"--gradient_accumulation_steps={config.training.gradient_accumulation_steps}",
        f"--learning_rate={config.training.learning_rate}",
        f"--warmup_steps={config.training.warmup_steps}",
        f"--eval_freq={config.training.eval_steps}",
        f"--save_freq={config.training.save_steps}",
        f"--log_freq={config.training.logging_steps}",
    ]
    
    if config.wandb.enabled:
        cmd.append("--wandb.enable=true")
        cmd.append(f"--wandb.project={config.wandb.project}")
    
    # Execute training
    print(f"\nCommand: {' '.join(cmd)}\n")
    os.system(" ".join(cmd))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLA for ESP32-CAM robot"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/smolvla_config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory"
    )
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = ESP32RobotConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = ESP32RobotConfig()
        print("Using default configuration")
    
    # Override with command line arguments
    if args.dataset:
        config.data.dataset_path = args.dataset
    if args.output:
        config.output_dir = args.output
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_wandb(config)
    
    # Train
    try:
        train_lerobot(config)
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
