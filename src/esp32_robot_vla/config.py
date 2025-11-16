# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyyaml",
#     "dataclasses",
# ]
# ///

"""Configuration for ESP32-CAM robot SmolVLA fine-tuning"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "lerobot/smolvla_base"
    vision_encoder: str = "siglip-so400m-patch14-384"
    action_dim: int = 2  # Left motor, Right motor
    state_dim: int = 6   # IMU: accel_xyz, gyro_xyz
    max_motor_speed: float = 255.0
    image_size: tuple = field(default_factory=lambda: (224, 224))


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration"""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_train_steps: int = 20000
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Hardware
    device: str = "cuda"
    mixed_precision: str = "bf16"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_path: str = "./data/robot_dataset"
    train_split: float = 0.9
    fps: int = 10
    max_episode_length: int = 1000
    augmentation: bool = True


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration"""
    project: str = "smolvla-esp32-robot"
    entity: Optional[str] = None
    run_name: str = "esp32_finetune_v1"
    log_model: bool = True
    enabled: bool = True


@dataclass
class ESP32RobotConfig:
    """Complete configuration for ESP32 robot fine-tuning"""
    output_dir: str = "./outputs/smolvla_esp32_robot"
    seed: int = 42
    
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ESP32RobotConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            output_dir=config_dict.get("output_dir", "./outputs/smolvla_esp32_robot"),
            seed=config_dict.get("seed", 42),
            model=ModelConfig(**config_dict.get("model", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            wandb=WandbConfig(**config_dict.get("wandb", {})),
        )
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            "output_dir": self.output_dir,
            "seed": self.seed,
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "wandb": self.wandb.__dict__,
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
