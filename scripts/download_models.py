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
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # For MVP, we'll use a smaller model that's actually available
        # TODO: For production, create fine-tuned models
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
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
        logger.info("You can manually download a model later")
    
    # Create placeholder for image model
    logger.info("Creating placeholder for image model...")
    image_model_path = models_dir / "image_detector.pth"
    
    # Save a minimal state dict as placeholder
    import torch
    placeholder_state = {
        "model_state_dict": {"placeholder": torch.tensor([1.0])},
        "version": "image_v1.0"
    }
    torch.save(placeholder_state, image_model_path)
    logger.info(f"Image model placeholder saved to {image_model_path}")
    
    logger.info("Model setup complete!")
    logger.info("\nNote: For production use, you should train and use proper models.")
    logger.info("The current models are placeholders for demonstration purposes.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())