""" 
Configuration Management for Blackwall. 
Handles Environment Variables, Default Settings, and Validation
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from functools import lru_cache
from pydantic import BaseSettings, validator, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LogLevel(str, Enum):
    """ Supported Log Levels """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(str, Enum):
    """ Supported Log Formats """
    JSON = "json"
    PLAIN = "plain"

class config(BaseSettings):
    """ Application Configuration """
    # Application settings
    app_name: str = "Blackwall"
    version: str = "0.1.0"
    
    # Logging configuration
    log_level: LogLevel = Field(LogLevel.INFO, env = "LOG_LEVEL")
    log_format: LogFormat = Field(LogFormat.JSON, env = "LOG_FORMAT")
    log_file: Optional[Path] = Field(None, env = "LOG_FILE")
    
    # Model paths
    text_model_path: Path = Field(
        Path("./src/blackwall/models/weights/text_detector.pth"),
        env="TEXT_MODEL_PATH"
    )
    image_model_path: Path = Field(
        Path("./src/blackwall/models/weights/image_detector.pth"),
        env="IMAGE_MODEL_PATH"
    )
    
    # Performance settings
    max_workers: int = Field(4, env = "MAX_WORKERS", ge = 1, le = 16)
    batch_size: int = Field(32, env = "BATCH_SIZE", ge = 1, le = 256)
    max_file_size_mb: int = Field(100, env = "MAX_FILE_SIZE_MB", ge = 1, le = 1000)
    device: str = Field("cpu", env = "DEVICE")  # cpu, cuda, mps
    
    # Cache settings
    enable_cache: bool = Field(True, env = "ENABLE_CACHE")
    cache_dir: Path = Field(
        Path.home() / ".blackwall" / "cache",
        env="CACHE_DIR"
    )
    
    # Timeout settings
    detection_timeout: int = Field(30, env = "DETECTION_TIMEOUT", ge = 5, le = 300)
    
    # Output settings
    verbose: bool = Field(False, env = "VERBOSE")
    output_format: str = Field("plain", env = "OUTPUT_FORMAT")  # plain, json
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("text_model_path", "image_model_path")
    def validate_model_paths(cls, v: Path) -> Path:
        """ Ensure Model Paths Exist or Can Be Created """
        if not v.exists():
            v.parent.mkdir(parents = True, exist_ok = True)
        return v
    
    @validator("cache_dir")
    def create_cache_dir(cls, v: Path) -> Path:
        """ Create Cache Directory if it Doesn't Exist """
        v.mkdir(parents = True, exist_ok = True)
        return v
    
    @validator("device")
    def validate_device(cls, v: str) -> str:
        """ Validate and Set Appropiate Device"""
        import torch
        
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        elif v == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return v
    
    @property
    def max_file_size_bytes(self) -> int:
        """ Get Max File Size in Bytes """
        return self.max_file_size_mb * 1024 * 1024
    
    def model_config_dict(self) -> Dict[str, Any]:
        """ Get  Model-Specific Configuration """
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }


@lru_cache()
def get_config() -> Config:
    """ Get cache Configuration Instance """
    return Config()


# Export commonly used functions
def get_model_path(model_type: str) -> Path:
    """ Get Path Specific Model """
    config = get_config()
    if model_type == "text":
        return config.text_model_path
    elif model_type == "image":
        return config.image_model_path
    else:
        raise ValueError(f"Unknown model type: {model_type}")