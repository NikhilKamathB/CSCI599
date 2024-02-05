###################################################################################################
# Data structure for the mesh must be defined in this file.
###################################################################################################

import logging
import numpy as np
from typing import DefaultDict, Tuple

logger = logging.getLogger(__name__)


class NpMesh:

    """
        Class to represent a mesh using numpy arrays.
    """

    def __init__(self, mesh: DefaultDict[str, list]) -> None:
        """
            Initialize the mesh object.
            Input parameters:
                - mesh: dictionary of objects associated with the 3D file
        """
        self.__LOG_PREFIX__ = "NpMesh()"
        if mesh is None or not mesh:
            logger.error(f"{self.__LOG_PREFIX__}: Invalid mesh object")
            raise ValueError("Invalid mesh object")
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the mesh object")
        self.raw_mesh = mesh
        self._build_mesh()

    def __str__(self) -> str:
        """
            String representation of the mesh.
        """
        return f"Vertices shape: {self.vertices.shape}\n Vertices: {self.vertices},\
            \n\nTexture coordinates shape: {self.texture_coordinates.shape},\nTexture coordinates: {self.texture_coordinates},\
            \n\nVertex normals shape: {self.vertex_normals.shape},\nVertex normals: {self.vertex_normals},\
            \n\nFaces shape: {self.faces.shape}\nFaces: {self.faces}"

    def _get_vertices(self) -> np.ndarray:
        """
            Get the vertices from the mesh.
            Output:
                - numpy array of vertices
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the vertices from the mesh")
        if 'v' not in self.raw_mesh:
            logger.error(
                f"{self.__LOG_PREFIX__}: Vertices not found in the mesh, vertices must be denoted by 'v' in the mesh default dict")
            raise ValueError("Vertices not found in the mesh, vertices must be denoted by 'v' in the mesh default dict")
        return self.convert_to_numpy_matrix(self.raw_mesh['v'])

    def _get_texture_coordinates(self) -> np.ndarray:
        """
            Get the texture coordinates from the mesh.
            Output:
                - numpy array of texture coordinates
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the texture coordinates from the mesh")
        if 'vt' not in self.raw_mesh:
            logger.error(
                f"{self.__LOG_PREFIX__}: Texture coordinates not found in the mesh, texture coordinates must be denoted by 'vt' in the mesh default dict")
            raise ValueError(
                "Texture coordinates not found in the mesh, texture coordinates must be denoted by 'vt' in the mesh default dict")
        return self.convert_to_numpy_matrix(self.raw_mesh['vt'])
    
    def _get_vertex_normals(self) -> np.ndarray:
        """
            Get the vertex normals from the mesh.
            Output:
                - numpy array of vertex normals
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the vertex normals from the mesh")
        if 'vn' not in self.raw_mesh:
            logger.error(
                f"{self.__LOG_PREFIX__}: Vertex normals not found in the mesh, vertex normals must be denoted by 'vn' in the mesh default dict")
            raise ValueError(
                "Vertex normals not found in the mesh, vertex normals must be denoted by 'vn' in the mesh default dict")
        return self.convert_to_numpy_matrix(self.raw_mesh['vn'])
    
    def _get_faces(self) -> np.ndarray:
        """
            Get the faces from the mesh.
            Output:
                - numpy array of faces
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the faces from the mesh")
        if 'f' not in self.raw_mesh:
            logger.error(
                f"{self.__LOG_PREFIX__}: Faces not found in the mesh, faces must be denoted by 'f' in the mesh default dict")
            raise ValueError("Faces not found in the mesh, faces must be denoted by 'f' in the mesh default dict")
        return self.convert_to_numpy_matrix(self.raw_mesh['f'])
    
    def _get_face_vertices(self) -> np.ndarray:
        """
            Get the face vertices from the mesh.
            Output:
                - numpy array of face vertices
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face vertices from the mesh")
        return self.faces[:, :, 0]
    
    def _get_face_texture_coordinates(self) -> np.ndarray:
        """
            Get the face texture coordinates from the mesh.
            Output:
                - numpy array of face texture coordinates
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face texture coordinates from the mesh")
        return self.faces[:, :, 1]
    
    def _get_face_vertex_normals(self) -> np.ndarray:
        """
            Get the face vertex normals from the mesh.
            Output:
                - numpy array of face vertex normals
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face vertex normals from the mesh")
        return self.faces[:, :, 2]
    
    def _build_mesh(self) -> None:
        """
            Build the mesh from the raw mesh data.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Building the mesh from the raw mesh data")
        self.vertices = self._get_vertices()
        self.texture_coordinates = self._get_texture_coordinates()
        self.vertex_normals = self._get_vertex_normals()
        self.faces = self._get_faces()
    
    def _get_face_items(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Get the face items from the mesh.
            Output:
                - numpy array of face items
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face items of the mesh")
        return self._get_face_vertices(), self._get_face_texture_coordinates(), self._get_face_vertex_normals()

    @staticmethod
    def convert_to_numpy_matrix(data: list) -> np.ndarray:
        """
            Converts list to a numpy array.
            Input parameters:
                - data: list of items
            Output:
                - numpy array of dtype float32
        """
        return np.asarray(data, dtype=np.float32)