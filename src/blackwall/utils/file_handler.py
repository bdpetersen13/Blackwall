"""
File Handling Utilities for Blackwall
Provides File Type Detection, Validation, and Processing Utilities
"""
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import hashlib
import magic
from blackwall.config import get_config
from blackwall.utils.logger import get_logger
from blackwall.utils.exceptions import FileNotSupportedError


logger = get_logger(__name__)
config = get_config()


# File type mappings
TEXT_EXTENSIONS = {".txt", ".md", ".docx", ".doc", ".pdf", ".rtf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}

TEXT_MIMETYPES = {
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/pdf",
    "application/rtf"
}

IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff"
}

VIDEO_MIMETYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
    "video/x-flv",
    "video/x-ms-wmv"
}


class FileHandler:
    """ Handles File Type Detection and Validation """
    
    def __init__(self):
        self.config = config
        self.logger = logger
        
        # Initialize python-magic
        try:
            self.magic = magic.Magic(mime = True)
        except Exception as e:
            self.logger.warning(
                "magic_init_failed",
                error = str(e),
                message = "Falling back to mimetypes module"
            )
            self.magic = None
    
    def detect_file_type(self, file_path: Path) -> str:
        """
        Detect file type (text, image, or video).
        
        Args:
            file_path: Path to the file
        
        Returns:
            File type string: "text", "image", or "video"
        
        Raises:
            FileNotSupportedError: If file type is not supported
        """
        file_path = Path(file_path)
        
        # First try extension-based detection
        extension = file_path.suffix.lower()
        
        if extension in TEXT_EXTENSIONS:
            return "text"
        elif extension in IMAGE_EXTENSIONS:
            return "image"
        elif extension in VIDEO_EXTENSIONS:
            return "video"
        
        # Fall back to MIME type detection
        mime_type = self._get_mime_type(file_path)
        
        if mime_type in TEXT_MIMETYPES:
            return "text"
        elif mime_type in IMAGE_MIMETYPES:
            return "image"
        elif mime_type in VIDEO_MIMETYPES:
            return "video"
        
        # If we can't determine the type, raise an error
        raise FileNotSupportedError(
            str(file_path),
            mime_type
        )
    
    def _get_mime_type(self, file_path: Path) -> Optional[str]:
        """Get MIME Type of File"""
        try:
            # Try python-magic first
            if self.magic:
                return self.magic.from_file(str(file_path))
            
            # Fall back to mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type
        
        except Exception as e:
            self.logger.warning(
                "mime_detection_failed",
                file_path = str(file_path),
                error = str(e)
            )
            return None
    
    def validate_file(
        self,
        file_path: Path,
        allowed_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate file for processing.
        
        Args:
            file_path: Path to the file
            allowed_types: List of allowed file types
        
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        validation_result = {
            "valid": True,
            "file_path": file_path,
            "exists": file_path.exists(),
            "is_file": file_path.is_file() if file_path.exists() else False,
            "size": 0,
            "type": None,
            "errors": []
        }
        
        # Check existence
        if not validation_result["exists"]:
            validation_result["valid"] = False
            validation_result["errors"].append("File does not exist")
            return validation_result
        
        # Check if it's a file
        if not validation_result["is_file"]:
            validation_result["valid"] = False
            validation_result["errors"].append("Path is not a file")
            return validation_result
        
        # Check size
        file_size = file_path.stat().st_size
        validation_result["size"] = file_size
        
        if file_size == 0:
            validation_result["valid"] = False
            validation_result["errors"].append("File is empty")
        elif file_size > self.config.max_file_size_bytes:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"File too large ({file_size / 1024 / 1024:.2f}MB > "
                f"{self.config.max_file_size_mb}MB)"
            )
        
        # Check type
        try:
            file_type = self.detect_file_type(file_path)
            validation_result["type"] = file_type
            
            if allowed_types and file_type not in allowed_types:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"File type '{file_type}' not in allowed types: "
                    f"{', '.join(allowed_types)}"
                )
        
        except FileNotSupportedError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
        
        return validation_result
    
    def get_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use
        
        Returns:
            Hex digest of the file hash
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """ Get Comprehensive File Information """
        file_path = Path(file_path)
        stat = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_path.suffix,
            "size": stat.st_size,
            "size_mb": stat.st_size / 1024 / 1024,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "type": self.detect_file_type(file_path),
            "mime_type": self._get_mime_type(file_path),
            "hash": self.get_file_hash(file_path)
        }


# Singleton instance
file_handler = FileHandler()


# Utility functions
def detect_file_type(file_path: Path) -> str:
    """ Detect File Type (Text, Image, or Video) """
    return file_handler.detect_file_type(file_path)


def validate_file(
    file_path: Path,
    allowed_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """ Validate File for Processing """
    return file_handler.validate_file(file_path, allowed_types)


def get_file_info(file_path: Path) -> Dict[str, Any]:
    """ Get Comprehensive File Information """
    return file_handler.get_file_info(file_path)