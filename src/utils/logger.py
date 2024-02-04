###################################################################################################
# Define a logger for the project here. This logger will be used to log all the information
# and errors during the execution of the project.
###################################################################################################

import os
import logging
from datetime import datetime

def __setup_logger__(log_dir: str = "/logs", level: int = logging.DEBUG) -> None:
    """
    Setup the logger for the project. The logs will be saved in the specified directory.
    Input parameters:
        - log_dir: directory where the logs will be saved.
    Output:
        - None
    """
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist!"
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        level=level, 
        format="[%(levelname)s] || %(asctime)s || %(name)s || %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )