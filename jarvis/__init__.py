"""Jarvis: a personal assistant with tools, memory and web learning.

Standalone by default: no API key, no network model. An OpenAI-compatible backend is optional (JARVIS_BACKEND=openai).
"""

from .agent import Assistant
from .config import Settings, load_settings
from .knowledge import KnowledgeBase, chunk_text

__all__ = ["Assistant", "KnowledgeBase", "Settings", "chunk_text", "load_settings"]
