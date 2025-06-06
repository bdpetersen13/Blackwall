"""
Model Loading Utilities for Blackwall
Handles Downloading, Caching, and Loading of detection Models
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import hashlib
import requests
from tqdm import tqdm
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from blackwall.config import get_config
from blackwall.utils.logger import get_logger
from blackwall.utils.exceptions import ModelLoadError


logger = get_logger(__name__)
config = get_config()


# Model registry with download URLs and metadata
MODEL_REGISTRY = {
    "text_detector_v1": {
        "type": "transformers",
        "model_name": "roberta-base-openai-detector",
        "url": "https://huggingface.co/roberta-base-openai-detector",
        "size_mb": 450,
        "hash": "sha256:xxxxx",  # Would be actual hash in production
        "description": "RoBERTa-based text detector fine-tuned on GPT outputs"
    },
    "image_detector_v1": {
        "type": "onnx",
        "model_name": "resnet50_gan_detector",
        "url": "https://example.com/models/image_detector_v1.onnx",
        "size_mb": 100,
        "hash": "sha256:xxxxx",
        "description": "ResNet50-based image detector for GAN/diffusion detection"
    }
}


class ModelLoader:
    """ Handles Modeling Loading and Management """
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self.device = self._get_device()
        self._model_cache = {}
    
    def _get_device(self) -> torch.device:
        """ Get Appropiate Device for Model Execution """
        device_str = self.config.device
        
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_transformers_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ) -> tuple:
        """
        Load a transformers model and tokenizer.
        
        Args:
            model_name: HuggingFace model name or path
            model_path: Local path to model weights
            cache_dir: Directory for caching models
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Use cache if model already loaded
            cache_key = f"transformers_{model_name}"
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]
            
            self.logger.info(
                "loading_transformers_model",
                model_name = model_name,
                device = str(self.device)
            )
            
            # Set cache directory
            if cache_dir is None:
                cache_dir = self.config.cache_dir / "models"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir = cache_dir,
                local_files_only = model_path is not None
            )
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path or model_name,
                cache_dir = cache_dir,
                local_files_only = model_path is not None
            )
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            
            # Cache the loaded model
            self._model_cache[cache_key] = (model, tokenizer)
            
            self.logger.info(
                "model_loaded",
                model_name = model_name,
                parameters = sum(p.numel() for p in model.parameters())
            )
            
            return model, tokenizer
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load transformers model {model_name}: {str(e)}",
                details={"model_name": model_name, "error": str(e)}
            )
    
    def load_onnx_model(
        self,
        model_path: Path,
        providers: Optional[list] = None
    ) -> ort.InferenceSession:
        """
        Load an ONNX model.
        
        Args:
            model_path: Path to ONNX model file
            providers: ONNX Runtime providers
        
        Returns:
            ONNX Runtime InferenceSession
        """
        try:
            # Use cache if model already loaded
            cache_key = f"onnx_{model_path}"
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]
            
            self.logger.info(
                "loading_onnx_model",
                model_path = str(model_path)
            )
            
            # Set providers based on device
            if providers is None:
                if self.device.type == "cuda":
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
            
            # Load model
            session = ort.InferenceSession(
                str(model_path),
                providers = providers
            )
            
            # Cache the session
            self._model_cache[cache_key] = session
            
            self.logger.info(
                "onnx_model_loaded",
                model_path = str(model_path),
                providers = providers
            )
            
            return session
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load ONNX model {model_path}: {str(e)}",
                details={"model_path": str(model_path), "error": str(e)}
            )
    
    def download_model(
        self,
        model_key: str,
        destination: Optional[Path] = None
    ) -> Path:
        """
        Download a model from the registry.
        
        Args:
            model_key: Key in MODEL_REGISTRY
            destination: Where to save the model
        
        Returns:
            Path to downloaded model
        """
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}")
        
        model_info = MODEL_REGISTRY[model_key]
        
        if destination is None:
            destination = self.config.cache_dir / "models" / f"{model_key}.bin"
        
        destination.parent.mkdir(parents = True, exist_ok = True)
        
        # Check if already downloaded
        if destination.exists():
            self.logger.info(
                "model_already_downloaded",
                model_key = model_key,
                path=str(destination)
            )
            return destination
        
        self.logger.info(
            "downloading_model",
            model_key = model_key,
            url = model_info["url"],
            size_mb = model_info["size_mb"]
        )
        
        # Download with progress bar
        response = requests.get(model_info["url"], stream = True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(destination, 'wb') as f:
            with tqdm(total = total_size, unit = 'iB', unit_scale = True) as pbar:
                for chunk in response.iter_content(block_size):
                    pbar.update(len(chunk))
                    f.write(chunk)
        
        self.logger.info(
            "model_downloaded",
            model_key = model_key,
            path = str(destination)
        )
        
        return destination


# Singleton instance
model_loader = ModelLoader()