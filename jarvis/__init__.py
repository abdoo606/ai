"""Jarvis: a personal assistant with tools, memory and web learning.

Works offline in a basic mode. Set JARVIS_LLM_API_KEY for a full language model.
"""

from .agent import Assistant
from .config import Settings, load_settings
from .knowledge import KnowledgeBase, chunk_text

__all__ = ["Assistant", "KnowledgeBase", "Settings", "chunk_text", "load_settings"]
