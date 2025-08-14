import logging

# Configure basic logging for the package
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__all__ = ["settings"]
