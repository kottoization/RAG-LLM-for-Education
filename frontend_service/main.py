"""Entry point for launching the Gradio frontend."""

import os
import sys

# Ensure the repository root is on the path when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend_service.interface import launch_gradio


def main() -> None:
    launch_gradio()


if __name__ == "__main__":
    main()
