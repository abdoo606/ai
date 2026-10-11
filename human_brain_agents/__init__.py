"""Brain-inspired multi-agent system.

Each agent models a group of neurons (a brain region) with one main job.
Agents exchange sparse, delayed messages along a connectome-like graph and
spend a limited energy budget on every step.
"""

from .core import Agent, Brain, Message
from .regions import build_default_brain

__all__ = ["Agent", "Brain", "Message", "build_default_brain"]
