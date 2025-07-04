# cli_generate.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from run_batch_of_slides import generate_help_text


if __name__ == "__main__":
    help_text = generate_help_text()
    print(help_text)
