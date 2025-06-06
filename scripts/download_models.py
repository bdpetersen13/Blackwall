"""
Script to Download Pre-Trained Models for Blackwall
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blackwall.models.model_loader import model_loader, MODEL_REGISTRY
from blackwall.config import get_config
from blackwall.utils.logger import get_logger


def main():
    """ Download All Required Models """
    logger = get_logger("download_models")
    config = get_config()
    
    logger.info("Starting model download")
    
    # Create models directory
    models_dir = Path("src/blackwall/models/weights")
    models_dir.mkdir(parents = True, exist_ok = True)
    
    # Download text model
    try:
        logger.info("Downloading text detection model...")
        # TODO: In production, this would download my fine-tuned model repository
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "roberta-base-openai-detector"
        
        # Download and save locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Save to local path
        save_path = models_dir / "text_detector"
        save_path.mkdir(exist_ok = True)
        
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        logger.info(f"Text model saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to download text model: {e}")
        return 1
    
    logger.info("All models downloaded successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())