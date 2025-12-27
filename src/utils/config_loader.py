"""Configuration loader utility for the RAG system."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls, config_path: str = "config.yaml") -> 'ConfigLoader':
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'embeddings.model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return the full configuration dictionary."""
        return self._config
    
    def reload(self, config_path: str = "config.yaml") -> None:
        """Reload configuration from file."""
        self._load_config(config_path)


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Factory function to get ConfigLoader instance."""
    return ConfigLoader(config_path)
