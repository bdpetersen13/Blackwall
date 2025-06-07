"""
Unit Tests for Text Detector
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from blackwall.detectors.text import TextDetector
from blackwall.utils.exceptions import InvalidInputError


class TestTextDetector:
    """ Test Text Detector Funcitonality """
    
    def test_detector_properties(self):
        """ Test Detector Properties """
        detector = TextDetector()
        
        assert detector.detector_type == "text"
        assert ".txt" in detector.supported_extensions
        assert ".md" in detector.supported_extensions
        assert ".docx" in detector.supported_extensions
    
    def test_text_cleaning(self):
        """Test Text Cleaning Functionality """
        detector = TextDetector()
        
        # Test whitespace cleaning
        dirty_text = "This   has    excessive\n\n\nwhitespace"
        clean_text = detector._clean_text(dirty_text)
        assert "excessive whitespace" in clean_text
        assert "\n\n\n" not in clean_text
    
    def test_feature_extraction(self):
        """ Test Feature Extraction """
        detector = TextDetector()
        
        text = "This is a test. This is only a test. Testing is important."
        features = detector._extract_text_features(text)
        
        assert "word_count" in features
        assert "sentence_count" in features
        assert "vocabulary_diversity" in features
        assert features["word_count"] == 12
        assert features["sentence_count"] == 3
    
    def test_ai_phrase_detection(self):
        """Test AI Phrase Counting """
        detector = TextDetector()
        
        text = "It's important to note that in today's world, we must be careful."
        count = detector._count_ai_phrases(text)
        assert count >= 2  # "it's important to note" and "in today's world"
    
    def test_repetition_rate(self):
        """Test Repetition Rate Calculation """
        detector = TextDetector()
        
        # High repetition
        words = ["the", "cat", "sat", "the", "cat", "sat", "the", "cat", "sat"]
        rate = detector._calculate_repetition_rate(words)
        assert rate > 0.5
        
        # Low repetition
        words = ["unique", "words", "without", "any", "repetition", "here"]
        rate = detector._calculate_repetition_rate(words)
        assert rate < 0.1
    
    @patch('blackwall.detectors.text.model_loader')
    def test_model_loading(self, mock_loader):
        """ Test Model Loading """
        detector = TextDetector()
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_loader.load_transformers_model.return_value = (mock_model, mock_tokenizer)
        
        detector.load_model()
        
        assert detector._model_loaded
        assert detector.model is not None
        assert detector.tokenizer is not None
    
    def test_invalid_file_handling(self, temp_dir: Path):
        """ Test Handling Invalid Files """
        detector = TextDetector()
        
        # Empty file
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        with pytest.raises(InvalidInputError):
            detector.preprocess(empty_file)
    
    def test_encoding_detection(self, temp_dir: Path):
        """ Testr File Encoding Detection """
        detector = TextDetector()
        
        # Create file with specific encoding
        file_path = temp_dir / "encoded.txt"
        text = "Test with special characters: café, naïve"
        file_path.write_text(text, encoding = "utf-8")
        
        read_text = detector._read_text_file(file_path)
        assert "café" in read_text
        assert "naïve" in read_text