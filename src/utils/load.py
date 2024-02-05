###################################################################################################
# Loading of any kind of 3D file should be done here.
# Supported file types include:
#    - .obj
###################################################################################################

import os
import logging
from collections import defaultdict
from typing import DefaultDict, Union
from src.base.ds_mesh import NpMesh
from src.utils.utils import floatify, intify

logger = logging.getLogger(__name__)


def load_obj(file_path: str) -> DefaultDict[str, list]:
    """
    Load a Wavefront .obj file into a dictionary of associated objects.
    Input parameters:
        - file_path: path to the .obj file
    Output:
        - defaultdict of objects associated with the .obj file
    """
    __LOG_PREFIX__ = "load_obj()"
    logger.info(f"{__LOG_PREFIX__}: Loading .obj file: {file_path}")
    try:
        mesh = defaultdict(list)
        with open(file_path, 'r') as file:
            for line in file:
                item = line.strip().split()
                # Skip empty lines
                if item is None or not item:
                    continue
                # Read the vertices
                if item[0] == 'v':
                    mesh['v'].append(floatify(item[1:]))
                # Read the texture coordinates
                elif item[0] == 'vt':
                    mesh['vt'].append(floatify(item[1:]))
                # Read the normals
                elif item[0] == 'vn':
                    mesh['vn'].append(floatify(item[1:]))
                # Read the faces
                elif item[0] == 'f':
                    mesh['f'].append([intify(i.split('/')) for i in item[1:]])
    except Exception as e:
        logger.error(f"{__LOG_PREFIX__}: Error while loading .obj file: {file_path}")
        raise e
    return mesh


def load_3d_file(file_path: str, base_ds: str = "np") -> Union[NpMesh, None]:
    """
    Load a 3D file into a dictionary of associated objects.
    Input parameters:
        - file_path: path to the 3D file
        - base_ds: base data structure to use for the mesh
    Output:
        - dictionary of objects associated with the 3D file
    """
    __LOG_PREFIX__ = "load_3d_file()"
    if not os.path.exists(file_path):
        logger.error(f"{__LOG_PREFIX__}: File does not exist: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")
    _file_ext = file_path.split('.')[-1]
    mesh = None
    if _file_ext == 'obj':
        mesh = load_obj(file_path)
    else:
        logger.error(f"{__LOG_PREFIX__}: Unsupported file format: {file_path}")
        raise NotImplementedError(f"Unsupported file format: {file_path}")
    if mesh is None:
        logger.error(f"{__LOG_PREFIX__}: Unable ot load the mesh as it is `None`")
        raise ValueError(f"Unable ot load the mesh as it is `None`")
    if base_ds == "np":
        return NpMesh(mesh)
    else:
        logger.error(f"{__LOG_PREFIX__}: Unsupported base data structure: {base_ds}")
        raise NotImplementedError(f"Unsupported base data structure: {base_ds}")