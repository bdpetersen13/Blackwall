"""
Image Detector Implementation for Blackwall
Detects AI-Generated Images Using CNN Models
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from blackwall.detectors.base import BaseDetector
from blackwall.detectors import register_detector
from blackwall.models.model_loader import model_loader
from blackwall.utils.logger import get_logger
from blackwall.utils.exceptions import ModelInferenceError, InvalidInputError


logger = get_logger(__name__)


class AIImageDetector(nn.Module):
    """ Custom CNN Model for AI Image Detection """
    
    def __init__(self, num_classes=2):
        super(AIImageDetector, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ImageDetector(BaseDetector):
    """ Detector for AI-Generated Images """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        self.model_version = "image_v1.0"
        self.image_size = (224, 224)
    
    @property
    def detector_type(self) -> str:
        return "image"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"]
    
    @property
    def model_path(self) -> Path:
        return self.config.image_model_path
    
    def load_model(self) -> None:
        """ Load the Image Detection Model """
        try:
            self.logger.info("loading_image_model")
            
            # Initialize model
            self.model = AIImageDetector(num_classes = 2)
            
            # Load weights if available
            if self.model_path.exists():
                self.logger.info(
                    "loading_model_weights",
                    path=str(self.model_path)
                )
                checkpoint = torch.load(
                    self.model_path,
                    map_location = model_loader.device
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model_version = checkpoint.get("version", "image_v1.0")
            else:
                self.logger.warning(
                    "model_weights_not_found",
                    path=str(self.model_path),
                    message="Using pre-trained ResNet50 weights"
                )
            
            # Move to device and set to eval
            self.model = self.model.to(model_loader.device)
            self.model.eval()
            
            # Setup image transforms
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self._model_loaded = True
            self.logger.info("image_model_loaded")
            
        except Exception as e:
            self.logger.error("image_model_load_failed", error = str(e))
            raise ModelInferenceError(f"Failed to load image model: {str(e)}")
    
    def preprocess(self, file_path: Path) -> Dict[str, Any]:
        """ Preprocess Image File for Detection"""
        try:
            # Load image
            image = Image.open(file_path).convert("RGB")
            
            # Extract metadata
            metadata = self._extract_image_metadata(image, file_path)
            
            # Check image validity
            if image.size[0] < 32 or image.size[1] < 32:
                raise InvalidInputError("Image too small for analysis")
            
            # Apply transforms for model
            tensor = self.transform(image)
            
            # Extract additional features
            features = self._extract_image_features(image)
            
            return {
                "tensor": tensor.unsqueeze(0),  # Add batch dimension
                "metadata": metadata,
                "features": features,
                "original_size": image.size
            }
            
        except Exception as e:
            self.logger.error(
                "image_preprocessing_failed",
                file_path = str(file_path),
                error = tr(e)
            )
            raise
    
    def run_inference(
        self,
        preprocessed_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """ Run Model Inference for Image Detection"""
        try:
            tensor = preprocessed_data["tensor"]
            features = preprocessed_data["features"]
            metadata = preprocessed_data["metadata"]
            
            # Move to device
            tensor = tensor.to(model_loader.device)
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim = 1)
                ai_probability = probabilities[0][1].item()  # Assuming label 1 is AI
            
            # Analyze patterns
            pattern_score = self._analyze_patterns(features)
            
            # Combine scores (70% model, 30% pattern analysis)
            final_probability = 0.7 * ai_probability + 0.3 * pattern_score
            
            # Build result metadata
            result_metadata = {
                "model_probability": round(ai_probability, 4),
                "pattern_score": round(pattern_score, 4),
                "image_size": preprocessed_data["original_size"],
                "format": metadata["format"],
                "mode": metadata["mode"],
                "features": features,
                "anomalies": self._detect_anomalies(features, ai_probability)
            }
            
            return final_probability, result_metadata
            
        except Exception as e:
            self.logger.error(
                "image_inference_failed",
                error = str(e),
                exc_info = True
            )
            raise ModelInferenceError(f"Image inference failed: {str(e)}")
    
    def _extract_image_metadata(
        self,
        image: Image.Image,
        file_path: Path
    ) -> Dict[str, Any]:
        """ Extract Image Meta Data"""
        metadata = {
            "format": image.format or "unknown",
            "mode": image.mode,
            "size": image.size,
            "info": image.info
        }
        
        # Try to get EXIF data
        try:
            from PIL.ExifTags import TAGS
            exifdata = image.getexif()
            if exifdata:
                exif = {}
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif[tag] = value
                metadata["exif"] = exif
        except:
            pass
        
        return metadata
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """ Extract Visual Feature from Images """
        # Convert to numpy array
        img_array = np.array(image)
        
        features = {}
        
        # Color statistics
        features["color_stats"] = self._calculate_color_statistics(img_array)
        
        # Frequency domain analysis
        features["frequency_features"] = self._analyze_frequency_domain(img_array)
        
        # Edge statistics
        features["edge_density"] = self._calculate_edge_density(img_array)
        
        # Noise characteristics
        features["noise_level"] = self._estimate_noise_level(img_array)
        
        return features
    
    def _calculate_color_statistics(self, img_array: np.ndarray) -> Dict[str, float]:
        """ Calculate Color Distribution Statistics """
        stats = {}
        
        # Calculate per-channel statistics
        for i, channel in enumerate(['red', 'green', 'blue']):
            channel_data = img_array[:, :, i]
            stats[f"{channel}_mean"] = float(np.mean(channel_data))
            stats[f"{channel}_std"] = float(np.std(channel_data))
            stats[f"{channel}_skew"] = float(self._calculate_skewness(channel_data))
        
        # Color diversity
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        stats["color_diversity"] = unique_colors / total_pixels
        
        return stats
    
    def _analyze_frequency_domain(self, img_array: np.ndarray) -> Dict[str, float]:
        """ Analyze Frequency Domain Characteristics """
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Calculate high frequency ratio
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Define regions
        center_size = min(rows, cols) // 8
        center_mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(center_mask, (ccol, crow), center_size, 1, -1)
        
        low_freq_sum = np.sum(magnitude_spectrum * center_mask)
        total_sum = np.sum(magnitude_spectrum)
        
        return {
            "high_freq_ratio": float(1 - (low_freq_sum / total_sum)) if total_sum > 0 else 0,
            "spectral_entropy": float(self._calculate_spectral_entropy(magnitude_spectrum))
        }
    
    def _calculate_edge_density(self, img_array: np.ndarray) -> float:
        """ Calculate Edge Density in the Image"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge pixel ratio
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        
        return float(edge_pixels / total_pixels)
    
    def _estimate_noise_level(self, img_array: np.ndarray) -> float:
        """ Estimate Noise Level in the Image """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Use Laplacian variance as noise estimate
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_estimate = laplacian.var()
        
        # Normalize to 0-1 range
        return float(min(noise_estimate / 1000, 1.0))
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """ Calculate Skewness of Data Distribution """
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_spectral_entropy(self, spectrum: np.ndarray) -> float:
        """ Calculate Entropy Frequency of Domain """
        # Normalize spectrum
        spectrum_norm = spectrum / np.sum(spectrum)
        
        # Calculate entropy
        spectrum_flat = spectrum_norm.flatten()
        spectrum_flat = spectrum_flat[spectrum_flat > 0]  # Remove zeros
        
        entropy = -np.sum(spectrum_flat * np.log2(spectrum_flat))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(spectrum_flat))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _analyze_patterns(self, features: Dict[str, Any]) -> float:
        """ Analyze Patterns to determin AI-Generated """
        score = 0.5  # Baseline
        
        # High frequency content (AI images often have specific frequency patterns)
        high_freq_ratio = features["frequency_features"]["high_freq_ratio"]
        if 0.3 < high_freq_ratio < 0.5:
            score += 0.15
        
        # Color diversity (AI images may have different color distributions)
        color_diversity = features["color_stats"]["color_diversity"]
        if color_diversity < 0.1:
            score += 0.1
        
        # Edge density (AI images often have characteristic edge patterns)
        edge_density = features["edge_density"]
        if 0.05 < edge_density < 0.15:
            score += 0.1
        
        # Low noise (AI images often have less natural noise)
        if features["noise_level"] < 0.1:
            score += 0.15
        
        return min(max(score, 0.0), 1.0)
    
    def _detect_anomalies(
        self,
        features: Dict[str, Any],
        model_probability: float
    ) -> List[str]:
        """ Detect Anomalies Indicating AI-Generation """
        anomalies = []
        
        if model_probability > 0.85:
            anomalies.append("Very high model confidence")
        
        if features["frequency_features"]["high_freq_ratio"] < 0.2:
            anomalies.append("Unusual frequency distribution")
        
        if features["color_stats"]["color_diversity"] < 0.05:
            anomalies.append("Low color diversity")
        
        if features["noise_level"] < 0.05:
            anomalies.append("Unnaturally low noise levels")
        
        if features["edge_density"] > 0.3:
            anomalies.append("Unusual edge patterns")
        
        return anomalies[:3]  # Return top 3


# Register the detector
register_detector("image", ImageDetector)