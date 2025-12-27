"""Logging utility for the RAG system."""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config_loader import get_config


class RAGLogger:
    """Custom logger for the RAG system."""
    
    _loggers: dict = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (typically module name)
            log_file: Optional log file path
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        config = get_config()
        log_level = config.get('logging.level', 'INFO')
        log_format = config.get(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            formatter = logging.Formatter(log_format)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (optional)
            if log_file:
                logs_dir = Path(config.get('paths.logs_dir', 'logs'))
                logs_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(logs_dir / log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Factory function to get a logger instance."""
    return RAGLogger.get_logger(name, log_file)
