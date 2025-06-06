"""
Base Detector Class Providing Common Functionality for all Detectors
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import time
from blackwall.config import get_config
from blackwall.utils.logger import get_logger, log_performance
from blackwall.utils.exceptions import DetectionError, ProcessingTimeoutError, CacheError



@dataclass
class DetectionResult:
    """ Result of a Detection Operation """
    
    file_path: Path
    file_type: str
    probability: float  # 0.0 to 1.0
    confidence: str  # "low", "medium", "high"
    is_ai_generated: bool
    processing_time: float  # seconds
    timestamp: datetime
    model_version: str
    metadata: Dict[str, Any]
    notes: List[str]
    
    def __post_init__(self):
        """ Validate and Process Fields after Initialization """
        # Ensure probability is in valid range
        self.probability = max(0.0, min(1.0, self.probability))
        
        # Set confidence based on probability
        if self.confidence is None:
            if self.probability < 0.3 or self.probability > 0.7:
                self.confidence = "high"
            elif 0.4 <= self.probability <= 0.6:
                self.confidence = "low"
            else:
                self.confidence = "medium"
        
        # Set is_ai_generated based on probability
        if self.is_ai_generated is None:
            self.is_ai_generated = self.probability >= 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """ Convert to Dictionary for Serialization """
        return {
            "file_path": str(self.file_path),
            "file_type": self.file_type,
            "probability": round(self.probability, 4),
            "confidence": self.confidence,
            "is_ai_generated": self.is_ai_generated,
            "processing_time": round(self.processing_time, 3),
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "metadata": self.metadata,
            "notes": self.notes
        }


class BaseDetector(ABC):
    """ Abstract Base Class for all Detectors """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(self.__class__.__name__)
        self._model = None
        self._model_loaded = False
        self.model_version = "unknown"
        
        # Initialize cache if enabled
        self._cache_dir = None
        if self.config.enable_cache:
            self._cache_dir = self.config.cache_dir / self.detector_type
            self._cache_dir.mkdir(parents = True, exist_ok = True)
    
    @property
    @abstractmethod
    def detector_type(self) -> str:
        """ Return the Type of Detector (Text, Image, Video) """
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """ Return List of Supported File Extensions """
        pass
    
    @property
    @abstractmethod
    def model_path(self) -> Path:
        """ Return Path to the Model Weights"""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """ Load the Detection Model """
        pass
    
    @abstractmethod
    def preprocess(self, file_path: Path) -> Any:
        """ Preprocess the Input File for Detection """
        pass
    
    @abstractmethod
    def run_inference(self, preprocessed_data: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Run model inference on preprocessed data.
        
        Returns:
            Tuple of (probability, metadata)
        """
        pass
    
    def detect(self, file_path: Path) -> DetectionResult:
        """
        Main detection method.
        
        Args:
            file_path: Path to the file to analyze
        
        Returns:
            DetectionResult object
        """
        start_time = time.time()
        
        try:
            # Convert to Path object if string
            file_path = Path(file_path)
            
            # Log detection start
            self.logger.info(
                "detection_started",
                file_path=str(file_path),
                detector_type=self.detector_type
            )
            
            # Validate file
            self._validate_file(file_path)
            
            # Check cache if enabled
            if self.config.enable_cache:
                cached_result = self._get_cached_result(file_path)
                if cached_result:
                    self.logger.info(
                        "cache_hit",
                        file_path=str(file_path)
                    )
                    return cached_result
            
            # Load model if not loaded
            if not self._model_loaded:
                self.load_model()
            
            # Preprocess file
            preprocessed_data = self.preprocess(file_path)
            
            # Run inference with timeout
            probability, metadata = self._run_inference_with_timeout(
                preprocessed_data
            )
            
            # Create result
            processing_time = time.time() - start_time
            result = self._create_result(
                file_path = file_path,
                probability = probability,
                metadata = metadata,
                processing_time = processing_time
            )
            
            # Cache result if enabled
            if self.config.enable_cache:
                self._cache_result(file_path, result)
            
            # Log success
            self.logger.info(
                "detection_completed",
                file_path = str(file_path),
                probability = result.probability,
                processing_time = processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                "detection_failed",
                file_path = str(file_path),
                error = str(e),
                processing_time = processing_time,
                exc_info = True
            )
            raise DetectionError(
                f"Detection failed for {file_path}: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            )
    
    def _validate_file(self, file_path: Path) -> None:
        """ Validate the Input File """
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if it's a file
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {', '.join(self.supported_extensions)}"
            )
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size_bytes:
            from blackwall.utils.exceptions import FileTooLargeError
            raise FileTooLargeError(
                str(file_path),
                file_size,
                self.config.max_file_size_bytes
            )
    
    def _run_inference_with_timeout(
        self,
        preprocessed_data: Any
    ) -> Tuple[float, Dict[str, Any]]:
        """ Run Inference with Timeout Function """
        import signal
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        with ThreadPoolExecutor(max_workers = 1) as executor:
            future = executor.submit(self.run_inference, preprocessed_data)
            
            try:
                return future.result(timeout = self.config.detection_timeout)
            except TimeoutError:
                raise ProcessingTimeoutError(
                    self.config.detection_timeout,
                    "model_inference"
                )
    
    def _create_result(
        self,
        file_path: Path,
        probability: float,
        metadata: Dict[str, Any],
        processing_time: float
    ) -> DetectionResult:
        """ Create Detection Result Object """
        # Generate notes based on probability
        notes = []
        if probability >= 0.9:
            notes.append("Very high probability of AI-generation detected")
        elif probability >= 0.7:
            notes.append("High probability of AI-generation detected")
        elif probability >= 0.5:
            notes.append("Moderate probability of AI-generation detected")
        elif probability >= 0.3:
            notes.append("Low probability of AI-generation detected")
        else:
            notes.append("Very low probability of AI-generation detected")
        
        # Add confidence disclaimer
        notes.append(
            " NOTE: This is a Probability Estimate. False Positives and "
            " Negatives are Possible. Always Confirm with Additional Sources"
        )
        
        return DetectionResult(
            file_path = file_path,
            file_type = self.detector_type,
            probability = probability,
            confidence = None,  # Will be set in __post_init__
            is_ai_generated = None,  # Will be set in __post_init__
            processing_time = processing_time,
            timestamp = datetime.utcnow(),
            model_version = self.model_version,
            metadata = metadata,
            notes = notes
        )
    
    def _get_cache_key(self, file_path: Path) -> str:
        """ Generate Cache Key for File """
        # Include file stats in cache key
        stat = file_path.stat()
        key_data = f"{file_path}:{stat.st_size}:{stat.st_mtime}:{self.model_version}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, file_path: Path) -> Optional[DetectionResult]:
        """ Get Cached Result if Available """
        if not self._cache_dir:
            return None
        
        try:
            cache_key = self._get_cache_key(file_path)
            cache_file = self._cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    data = json.load(f)
                
                # Reconstruct DetectionResult
                return DetectionResult(
                    file_path = Path(data["file_path"]),
                    file_type = data["file_type"],
                    probability = data["probability"],
                    confidence = data["confidence"],
                    is_ai_generated = data["is_ai_generated"],
                    processing_time = data["processing_time"],
                    timestamp = datetime.fromisoformat(data["timestamp"]),
                    model_version = data["model_version"],
                    metadata = data["metadata"],
                    notes = data["notes"]
                )
        
        except Exception as e:
            self.logger.warning(
                " cache_read_failed ",
                error=str(e),
                file_path=str(file_path)
            )
            return None
    
    def _cache_result(self, file_path: Path, result: DetectionResult) -> None:
        """ Cache Detection Result """
        if not self._cache_dir:
            return
        
        try:
            cache_key = self._get_cache_key(file_path)
            cache_file = self._cache_dir / f"{cache_key}.json"
            
            with open(cache_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        
        except Exception as e:
            self.logger.warning(
                "cache_write_failed",
                error=str(e),
                file_path=str(file_path)
            )