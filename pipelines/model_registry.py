#!/usr/bin/env python3
"""
Model registry for forecasting pipeline.

Maintains a registry of available forecasting models.
"""

from typing import Dict, Callable, Any, Optional, List
import pandas as pd


class ModelRegistry:
    """
    Registry of available forecasting models.
    
    Models can be registered with a name and factory function.
    """

    def __init__(self):
        self._models: Dict[str, Callable] = {}

    def register(self, name: str, model_factory: Callable[[], Any]):
        """
        Register a model factory.
        
        Parameters:
        -----------
        name : str
            Model name
        model_factory : callable
            Factory function that returns a model instance
            Model must have: fit(production: pd.Series) and predict(...) methods
        """
        self._models[name] = model_factory

    def get(self, name: str):
        """
        Get a new instance of a registered model.
        
        Parameters:
        -----------
        name : str
            Model name
            
        Returns:
        --------
        Model instance
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(self._models.keys())}")
        return self._models[name]()

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._models


# Global registry instance
_default_registry = ModelRegistry()


def register_model(name: str, model_factory: Callable[[], Any]):
    """Register a model in the default registry."""
    _default_registry.register(name, model_factory)


def get_default_registry() -> ModelRegistry:
    """Get the default model registry."""
    return _default_registry

