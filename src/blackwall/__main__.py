""" Main Entry Point for Blackwall cli """
import sys

# Import detectors to register them
from blackwall.detectors import text, image, video
from blackwall.cli import main


if __name__ == "__main__":
    sys.exit(main())