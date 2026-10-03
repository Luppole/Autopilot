"""Free Model Router: sync OpenRouter's free models and route requests across them."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("free-model-router")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"
