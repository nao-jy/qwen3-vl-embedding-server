"""
Qwen3-VL-Embedding Server

OpenAI-compatible multimodal embedding server for Qwen3-VL-Embedding.
"""

__version__ = "1.0.0"
__author__ = "Phillip McGinty"

from .server import app, create_embeddings, health, root

__all__ = ["app", "create_embeddings", "health", "root"]
