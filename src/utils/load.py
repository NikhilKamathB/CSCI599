###################################################################################################
# Loading of any kind of 3D file should be done here.
# Supported file types include:
#    - .obj
###################################################################################################

import os
import logging

logger = logging.getLogger(__name__)

def load_obj(file_path: str) -> dict:
    """
    Load a Wavefront .obj file into a dictionary of associated objects.
    Input parameters:
        - file_path: path to the .obj file
    Output:
        - dictionary of objects associated with the .obj file
    """
    __LOG_PREFIX__ = "load_obj()"
    logger.info(f"{__LOG_PREFIX__}: Loading .obj file: {file_path}")

def load_3d_file(file_path: str) -> dict:
    """
    Load a 3D file into a dictionary of associated objects.
    Input parameters:
        - file_path: path to the 3D file
    Output:
        - dictionary of objects associated with the 3D file
    """
    __LOG_PREFIX__ = "load_3d_file()"
    if not os.path.exists(file_path):
        logger.error(f"{__LOG_PREFIX__}: File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")
    _file_ext = file_path.split('.')[-1]
    if _file_ext == 'obj':
        return load_obj(file_path)
    else:
        logger.error(f"{__LOG_PREFIX__}: Unsupported file format: {file_path}")
        raise NotImplementedError(f"Unsupported file format: {file_path}")