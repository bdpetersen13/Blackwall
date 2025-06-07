"""
Configuration Management for Blackwall
Handles Environmente Variables, Default Settings, and Validation
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
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
    """ Supported Log Formats"""
    JSON = "json"
    PLAIN = "plain"


class Config(BaseSettings):
    """ Application Configuration """
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
        extra = "ignore"
    )
    
    # Application settings
    app_name: str = "blackwall"
    version: str = "0.1.0"
    
    # Logging configuration
    log_level: LogLevel = Field(default = LogLevel.INFO, description = "Logging level")
    log_format: LogFormat = Field(default = LogFormat.JSON, description = "Log format")
    log_file: Optional[Path] = Field(default = None, description = "Log file path")
    
    # Model paths
    text_model_path: Path = Field(
        default=Path("./src/blackwall/models/weights/text_detector.pth"),
        description="Path to text detection model"
    )
    image_model_path: Path = Field(
        default=Path("./src/blackwall/models/weights/image_detector.pth"),
        description="Path to image detection model"
    )
    
    # Performance settings
    max_workers: int = Field(default = 4, ge = 1, le = 16, description = "Maximum worker threads")
    batch_size: int = Field(default = 32, ge = 1, le = 256, description = "Batch size for processing")
    max_file_size_mb: int = Field(default = 100, ge = 1, le = 1000, description = "Maximum file size in MB")
    device: str = Field(default = "cpu", description = "Device for computation (cpu, cuda, mps)")
    
    # Cache settings
    enable_cache: bool = Field(default = True, description = "Enable result caching")
    cache_dir: Path = Field(
        default_factory = lambda: Path.home() / ".blackwall" / "cache",
        description = "Cache directory"
    )
    
    # Timeout settings
    detection_timeout: int = Field(default = 30, ge = 5, le = 300, description = "Detection timeout in seconds")
    
    # Output settings
    verbose: bool = Field(default = False, description = "Enable verbose output")
    output_format: str = Field(default = "plain", description = "Output format")
    
    @field_validator("text_model_path", "image_model_path")
    @classmethod
    def validate_model_paths(cls, v: Path) -> Path:
        """ Ensures Model Path Exists or Can be Executed """
        if not v.exists():
            v.parent.mkdir(parents = True, exist_ok = True)
        return v
    
    @field_validator("cache_dir")
    @classmethod
    def create_cache_dir(cls, v: Path) -> Path:
        """ Create Cache Directory if it Doesn't Exist """
        v.mkdir(parents = True, exist_ok = True)
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """ Validate and Set Appropiate Device """
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
        """ Get Model-Speccific Configuration """
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }


@lru_cache()
def get_config() -> Config:
    """ Get Cached Configuration Instance """
    return Config()


# Export commonly used functions
def get_model_path(model_type: str) -> Path:
    """ Get Path for Specific Model Type """
    config = get_config()
    if model_type == "text":
        return config.text_model_path
    elif model_type == "image":
        return config.image_model_path
    else:
        raise ValueError(f"Unknown model type: {model_type}")