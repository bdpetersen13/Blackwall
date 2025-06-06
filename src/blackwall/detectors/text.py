"""
Text Detector Implementation for Blackwall
Detects AI-Generated Text Using Transformer Models
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import chardet
from blackwall.detectors.base import BaseDetector
from blackwall.detectors import register_detector
from blackwall.models.model_loader import model_loader
from blackwall.utils.logger import get_logger
from blackwall.utils.exceptions import ModelInferenceError, InvalidInputError


logger = get_logger(__name__)


class TextDetector(BaseDetector):
    """ Detector for AI-Generated Text """
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.max_length = 512
        self.model_version = "text_v1.0"
    
    @property
    def detector_type(self) -> str:
        return "text"
    
    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".md", ".docx", ".doc", ".pdf", ".rtf"]
    
    @property
    def model_path(self) -> Path:
        return self.config.text_model_path
    
    def load_model(self) -> None:
        """ Load the Text Detection Model """
        try:
            self.logger.info("loading_text_model")
            
            # For MVP, we'll use a pre-trained model
            # TODO: In production, create and load a fine-tuned model
            model_name = "roberta-base-openai-detector"
            
            # Try loading from local path first
            if self.model_path.exists():
                self.model, self.tokenizer = model_loader.load_transformers_model(
                    model_name,
                    model_path=self.model_path
                )
            else:
                # Download and load from HuggingFace
                self.model, self.tokenizer = model_loader.load_transformers_model(
                    model_name
                )
            
            self._model_loaded = True
            self.logger.info("text_model_loaded")
            
        except Exception as e:
            self.logger.error("text_model_load_failed", error=str(e))
            raise ModelLoadError(f"Failed to load text model: {str(e)}")
    
    def preprocess(self, file_path: Path) -> Dict[str, Any]:
        """ Preprocess Text File for Detection """
        try:
            # Read file content
            text = self._read_text_file(file_path)
            
            # Basic text cleaning
            text = self._clean_text(text)
            
            # Validate text
            if not text or len(text.strip()) < 10:
                raise InvalidInputError("Text too short for analysis")
            
            # Extract features
            features = self._extract_text_features(text)
            
            # Tokenize for model
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "text": text,
                "tokens": tokens,
                "features": features
            }
            
        except Exception as e:
            self.logger.error(
                "text_preprocessing_failed",
                file_path=str(file_path),
                error=str(e)
            )
            raise
    
    def run_inference(
        self,
        preprocessed_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """ Run Model Inference of Preprocessed Text """
        try:
            tokens = preprocessed_data["tokens"]
            features = preprocessed_data["features"]
            text = preprocessed_data["text"]
            
            # Move tokens to device
            tokens = {k: v.to(model_loader.device) for k, v in tokens.items()}
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(**tokens)
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs.logits, dim = -1)
                ai_probability = probabilities[0][1].item()  # Assuming label 1 is AI
            
            # Combine with feature-based analysis
            feature_score = self._calculate_feature_score(features)
            
            # Weighted combination (80% model, 20% features)
            final_probability = 0.8 * ai_probability + 0.2 * feature_score
            
            # Create metadata
            metadata = {
                "model_probability": round(ai_probability, 4),
                "feature_score": round(feature_score, 4),
                "text_length": len(text),
                "token_count": len(tokens["input_ids"][0]),
                "features": features,
                "top_indicators": self._get_top_indicators(features, ai_probability)
            }
            
            return final_probability, metadata
            
        except Exception as e:
            self.logger.error(
                "text_inference_failed",
                error = str(e),
                exc_info = True
            )
            raise ModelInferenceError(f"Text inference failed: {str(e)}")
    
    def _read_text_file(self, file_path: Path) -> str:
        """ Read Text from Various File Formats """
        extension = file_path.suffix.lower()
        
        try:
            if extension in [".txt", ".md"]:
                # Detect encoding
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected["encoding"] or "utf-8"
                
                return raw_data.decode(encoding)
            
            elif extension == ".docx":
                import docx
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            
            elif extension == ".pdf":
                import PyPDF2
                text = ""
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
            
            else:
                # Try reading as plain text
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
                    
        except Exception as e:
            raise InvalidInputError(f"Failed to read text file: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """ Clean and Normalize Text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        # Trim
        text = text.strip()
        
        return text
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """ Extract Linguistic Features from Text"""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        # Basic statistics
        features = {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
        }
        
        # Vocabulary diversity
        unique_words = set(w.lower() for w in words)
        features["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0
        
        # Repetition patterns
        features["word_repetition_rate"] = self._calculate_repetition_rate(words)
        
        # Common AI patterns
        features["ai_phrase_count"] = self._count_ai_phrases(text)
        
        # Punctuation patterns
        features["punctuation_diversity"] = self._calculate_punctuation_diversity(text)
        
        return features
    
    def _calculate_repetition_rate(self, words: List[str]) -> float:
        """ Calculate Word Repetition Rate """
        if len(words) < 10:
            return 0.0
        
        # Count 3-grams
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(" ".join(words[i:i+3]).lower())
        
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)
        
        return repeated_trigrams / len(trigrams) if trigrams else 0.0
    
    def _count_ai_phrases(self, text: str) -> int:
        """ Count Common GenAI Phrases"""
        ai_phrases = [
            "it's important to note",
            "it's worth noting",
            "in conclusion",
            "first and foremost",
            "in today's world",
            "in the modern era",
            "it's crucial to understand",
            "let's dive into",
            "that being said",
            "at the end of the day"
        ]
        
        text_lower = text.lower()
        count = sum(1 for phrase in ai_phrases if phrase in text_lower)
        
        return count
    
    def _calculate_punctuation_diversity(self, text: str) -> float:
        """ Calculate Punctuation Diversity Score """
        punctuation = ".,!?;:'-\""
        punct_counts = Counter(char for char in text if char in punctuation)
        
        if not punct_counts:
            return 0.0
        
        # Shannon entropy
        total = sum(punct_counts.values())
        entropy = -sum(
            (count/total) * np.log2(count/total) 
            for count in punct_counts.values()
        )
        
        # Normalize by max possible entropy
        max_entropy = np.log2(len(punctuation))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_feature_score(self, features: Dict[str, Any]) -> float:
        """ Calculate AI Probability Based on Features """
        score = 0.5  # Baseline
        
        # High repetition suggests AI
        if features["word_repetition_rate"] > 0.1:
            score += 0.2
        
        # Low vocabulary diversity suggests AI
        if features["vocabulary_diversity"] < 0.3:
            score += 0.1
        
        # AI phrases
        if features["ai_phrase_count"] > 2:
            score += 0.15
        
        # Low punctuation diversity
        if features["punctuation_diversity"] < 0.3:
            score += 0.05
        
        return min(max(score, 0.0), 1.0)
    
    def _get_top_indicators(
        self,
        features: Dict[str, Any],
        model_probability: float
    ) -> List[str]:
        """ Get Top Indicators of GenAI"""
        indicators = []
        
        if model_probability > 0.8:
            indicators.append("Model confidence very high")
        
        if features["word_repetition_rate"] > 0.1:
            indicators.append("High phrase repetition detected")
        
        if features["vocabulary_diversity"] < 0.3:
            indicators.append("Low vocabulary diversity")
        
        if features["ai_phrase_count"] > 2:
            indicators.append(f"Found {features['ai_phrase_count']} common AI phrases")
        
        return indicators[:3]  # Return top 3


# Register the detector
register_detector("text", TextDetector)