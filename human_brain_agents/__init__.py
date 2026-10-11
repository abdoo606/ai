"""Brain-inspired multi-agent model built on the HCP structural connectome.

Each brain region is an agent with a functional module, a limited energy
budget, delayed connections, and plastic excitatory links.
"""

from .core import Agent, Brain, Message
from .connectome import build_hcp_brain, load_structural
from .regions import make_agent, module_of, parse_label

__all__ = [
    "Agent", "Brain", "Message",
    "build_hcp_brain", "load_structural",
    "make_agent", "module_of", "parse_label",
]

# Backwards-compatible name used by the first prototype.
build_default_brain = build_hcp_brain
