##########################################################################################################
# Initialize the `src` package for the project
##########################################################################################################

# Import the required modules
import os
import logging
from .utils.logger import __setup_logger__
from .utils.load import load_3d_file

__LOGGING_DIR__ = "../logs"
__LOGGING_LEVEL__ = logging.DEBUG

if not os.path.exists(__LOGGING_DIR__):
    os.makedirs(__LOGGING_DIR__)
__setup_logger__(log_dir=__LOGGING_DIR__, level=__LOGGING_LEVEL__)

__all__ = [
    "load_3d_file"
]