"""
Pytest Configuration and Fixtures for Blackwall Tests
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Generator
from blackwall.config import Config
from blackwall.detectors.base import DetectionResult


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """ Create Temporary Directory for Tests """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config() -> Config:
    """ Create Test Configuration """
    return Config(
        log_level = "DEBUG",
        enable_cache = False,
        device = "cpu",
        max_file_size_mb = 10,
        detection_timeout = 10
    )


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """ Create a Sample Text File """
    file_path = temp_dir / "sample.txt"
    file_path.write_text(
        "This is a sample text file for testing. "
        "It contains multiple sentences to ensure proper analysis. "
        "The content should be long enough to trigger detection"
    )
    return file_path


@pytest.fixture
def sample_ai_text_file(temp_dir: Path) -> Path:
    """ Create a Sample AI-Generated Text File """
    file_path = temp_dir / "ai_sample.txt"
    file_path.write_text(
        "It's important to note that in today's world, artificial intelligence "
        "has become increasingly prevalent. First and foremost, we must understand "
        "that AI systems are designed to process information efficiently. "
        "It's worth noting that these systems can analyze vast amounts of data. "
        "In conclusion, the impact of AI on society continues to grow"
    )
    return file_path


@pytest.fixture
def sample_image_file(temp_dir: Path) -> Path:
    """ Create a Sample Image File """
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype = np.uint8)
    img = Image.fromarray(img_array)
    
    file_path = temp_dir / "sample.jpg"
    img.save(file_path)
    return file_path


@pytest.fixture
def mock_detection_result() -> DetectionResult:
    """ Create a Mock Detection Result """
    from datetime import datetime
    
    return DetectionResult(
        file_path = Path("test.txt"),
        file_type = "text",
        probability = 0.75,
        confidence = "high",
        is_ai_generated = True,
        processing_time = 1.23,
        timestamp = datetime.utcnow(),
        model_version = "test_v1.0",
        metadata = {"test": True},
        notes = ["Test note"]
    )