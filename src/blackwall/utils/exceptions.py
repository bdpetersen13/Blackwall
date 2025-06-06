"""
Custom Exceptionbs for Blackwall
Provides Specific Error Types for Better Error Handling and Debugging
"""
from typing import Optional, Dict, Any


class BlackwallError(Exception):
    """ Base Exception for all Blackwall Errors """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """ Convert Exception to Dictionary for Logging/API Responses """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(BlackwallError):
    """ Raised When COnfiguration is Invalid or Missing """
    pass


class ModelLoadError(BlackwallError):
    """ Raised When Model Loading Fails """
    pass


class DetectionError(BlackwallError):
    """ base Class for Detection-Related Errors """
    pass


class FileNotSupportedError(DetectionError):
    """ Raised When the File Type is Not Supported """
    
    def __init__(self, file_path: str, file_type: Optional[str] = None):
        message = f"File type not supported: {file_path}"
        if file_type:
            message += f" (detected type: {file_type})"
        
        super().__init__(
            message=message,
            details={"file_path": file_path, "file_type": file_type}
        )


class FileTooLargeError(DetectionError):
    """ Raised When the File Exceeds the Size Limit """
    
    def __init__(self, file_path: str, file_size: int, max_size: int):
        message = (
            f"File too large: {file_path} "
            f"({file_size / 1024 / 1024:.2f}MB > "
            f"{max_size / 1024 / 1024:.2f}MB)"
        )
        
        super().__init__(
            message=message,
            details={
                "file_path": file_path,
                "file_size": file_size,
                "max_size": max_size
            }
        )


class ProcessingTimeoutError(DetectionError):
    """ Raised When Detection is Taking Too Long """
    
    def __init__(self, timeout: int, operation: str):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout} seconds",
            details={"timeout": timeout, "operation": operation}
        )


class ModelInferenceError(DetectionError):
    """ Raised When the Model Inference Fails"""
    pass


class InvalidInputError(BlackwallError):
    """ Raised When Input Validation Fails """
    pass


class CacheError(BlackwallError):
    """ Raised When Cache Operations Fail """
    pass