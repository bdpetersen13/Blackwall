"""
Unit Tests for File Handler
"""
import pytest
from pathlib import Path
from blackwall.utils.file_handler import FileHandler, detect_file_type
from blackwall.utils.exceptions import FileNotSupportedError


class TestFileHandler:
    """ Test File Handler Functionality """
    
    def test_file_type_detection_by_extension(self, temp_dir: Path):
        """ Test File Type Detection by Extension """
        handler = FileHandler()
        
        # Test text file
        text_file = temp_dir / "test.txt"
        text_file.write_text("test")
        assert handler.detect_file_type(text_file) == "text"
        
        # Test image file
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"fake image data")
        assert handler.detect_file_type(image_file) == "image"
        
        # Test video file
        video_file = temp_dir / "test.mp4"
        video_file.write_bytes(b"fake video data")
        assert handler.detect_file_type(video_file) == "video"
    
    def test_unsupported_file_type(self, temp_dir: Path):
        """ Test Unsupported File Type Handling """
        handler = FileHandler()
        
        unknown_file = temp_dir / "test.xyz"
        unknown_file.write_text("unknown")
        
        with pytest.raises(FileNotSupportedError):
            handler.detect_file_type(unknown_file)
    
    def test_file_validation(self, temp_dir: Path):
        """ Test File Validation """
        handler = FileHandler()
        
        # Valid file
        valid_file = temp_dir / "valid.txt"
        valid_file.write_text("valid content")
        
        result = handler.validate_file(valid_file)
        assert result["valid"]
        assert result["exists"]
        assert result["is_file"]
        assert result["size"] > 0
        
        # Non-existent file
        result = handler.validate_file(temp_dir / "nonexistent.txt")
        assert not result["valid"]
        assert "does not exist" in result["errors"][0]
    
    def test_file_hash(self, temp_dir: Path):
        """ Test File Hash Calculation """
        handler = FileHandler()
        
        # Create file with known content
        file_path = temp_dir / "hash_test.txt"
        content = "Test content for hashing"
        file_path.write_text(content)
        
        # Calculate hash
        hash1 = handler.get_file_hash(file_path)
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
        
        # Same content should produce same hash
        hash2 = handler.get_file_hash(file_path)
        assert hash1 == hash2
        
        # Different content should produce different hash
        file_path.write_text("Different content")
        hash3 = handler.get_file_hash(file_path)
        assert hash1 != hash3