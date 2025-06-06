"""
Logging Configuration for Blackwall
Provides Structured Logging with Context and Performance Tracking
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.stdlib import LoggerFactory, BoundLogger
from blackwall.config import get_config


def setup_logging() -> None:
    """ Configureation for Structured Logging for Application """
    config = get_config()
    
    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt = "iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_context_processor,
        add_performance_processor,
    ]
    
    # Add appropriate renderer based on format
    if config.log_format == "json":
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors = processors,
        context_class = dict,
        logger_factory = LoggerFactory(),
        wrapper_class = BoundLogger,
        cache_logger_on_first_use = True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level.value),
    )
    
    # Add file handler if specified
    if config.log_file:
        add_file_handler(config.log_file, config.log_level.value)


def add_context_processor(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """ Add Contextual Information to Logs """
    config = get_config()
    
    # Add application context
    event_dict["app"] = config.app_name
    event_dict["version"] = config.version
    
    # Add timestamp if not present
    if "timestamp" not in event_dict:
        event_dict["timestamp"] = datetime.utcnow().isoformat()
    
    return event_dict


def add_performance_processor(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """ Add Performance Metrics to Logs"""
    import psutil
    
    # Add memory usage
    process = psutil.Process()
    event_dict["memory_mb"] = process.memory_info().rss / 1024 / 1024
    
    # Add CPU usage (if available)
    try:
        event_dict["cpu_percent"] = process.cpu_percent(interval = 0.1)
    except:
        pass
    
    return event_dict


def add_file_handler(
    log_file: Path,
    log_level: str,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """ Add Rotating File Handler to Root Log """
    from logging.handlers import RotatingFileHandler
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents = True, exist_ok = True)
    
    # Create rotating file handler
    handler = RotatingFileHandler(
        log_file,
        maxBytes = max_bytes,
        backupCount = backup_count,
        encoding="utf-8"
    )
    
    # Set formatter
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s"}'
    )
    handler.setFormatter(formatter)
    handler.setLevel(getattr(logging, log_level))
    
    # Add to root logger
    logging.getLogger().addHandler(handler)


def get_logger(name: Optional[str] = None) -> BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (defaults to module name)
    
    Returns:
        Configured logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "blackwall")
        else:
            name = "blackwall"
    
    return structlog.get_logger(name)


class LogContext:
    """ Context Manager for Adding Temporary Log Context """
    
    def __init__(self, logger: BoundLogger, **kwargs):
        self.logger = logger
        self.context = kwargs
        self.original_context = None
    
    def __enter__(self):
        self.original_context = self.logger._context
        self.logger = self.logger.bind(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        if self.original_context is not None:
            self.logger._context = self.original_context


# Performance logging decorator
def log_performance(func):
    """ Decorator to Log Function Performance Metrics """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            logger.info(
                "function_completed",
                function=func.__name__,
                duration_seconds=duration,
                success=True
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                "function_failed",
                function=func.__name__,
                duration_seconds = duration,
                error=str(e),
                exc_info=True
            )
            raise
    
    return wrapper


# Initialize logging when module is imported
setup_logging()