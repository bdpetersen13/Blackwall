"""
Detector Modules for Blackwall
"""
from typing import Dict, Type
from pathlib import Path
from blackwall.detectors.base import BaseDetector, DetectionResult
from blackwall.utils.file_handler import detect_file_type


# Detector registry - will be populated when specific detectors are imported
DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register_detector(file_type: str, detector_class: Type[BaseDetector]) -> None:
    """ Register a Detector for a Specific File Type """
    DETECTOR_REGISTRY[file_type] = detector_class


def get_detector(file_type: str) -> BaseDetector:
    """
    Get appropriate detector instance for file type.
    
    Args:
        file_type: Type of file (text, image, video)
    
    Returns:
        Detector instance
    
    Raises:
        ValueError: If no detector is registered for file type
    """
    if file_type not in DETECTOR_REGISTRY:
        raise ValueError(
            f"No detector registered for file type: {file_type}. "
            f"Available types: {list(DETECTOR_REGISTRY.keys())}"
        )
    
    detector_class = DETECTOR_REGISTRY[file_type]
    return detector_class()


def detect_file(file_path: Path) -> DetectionResult:
    """
    Detect if file is AI-generated.
    
    This is a convenience function that automatically selects
    the appropriate detector based on file type.
    
    Args:
        file_path: Path to the file to analyze
    
    Returns:
        DetectionResult object
    """
    # Detect file type
    file_type = detect_file_type(file_path)
    
    # Get appropriate detector
    detector = get_detector(file_type)
    
    # Run detection
    return detector.detect(file_path)


__all__ = [
    "BaseDetector",
    "DetectionResult",
    "register_detector",
    "get_detector",
    "detect_file"
]