""" Blackwall - GenAI Detection Tool """
from blackwall.config import get_config

__version__ = get_config().version
__all__ = ["__version__"]