##########################################################################################################
# Any remeshing angorithms should be implemented here.
# Algorithms implemented:
#   - Loop Subdivision
##########################################################################################################

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Remesh:
    
    """
        Class to represent remeshing algorithms.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """
        Initialize the remesh object.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, note that the faces should be triangulated
        """
        self.__LOG_PREFIX__ = "Remesh()"
        if vertices is None or not vertices or faces is None or not faces:
            logger.error(f"{self.__LOG_PREFIX__}: Invalid vertices or faces")
            raise ValueError("Invalid vertices or faces")
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the remesh object")
        self.vertices = vertices
        self.faces = faces

    def loop_subdivision(self, iterations=1):
        """
        Apply Loop subdivision to the input mesh for the specified number of iterations.
        :param iterations: number of iterations
        :return: mesh after subdivision
        """
        logger.info(f"{self.__LOG_PREFIX__}: Applying Loop subdivision to the input mesh")
        return