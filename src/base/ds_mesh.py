###################################################################################################
# Data structure for the mesh must be defined in this file.
###################################################################################################

import os
import logging
import numpy as np
import src.utils.utils as utils
from typing import DefaultDict, Tuple
from src.remesh import Remesh

logger = logging.getLogger(__name__)


class NpMesh:

    """
        Class to represent a mesh using numpy arrays.
    """

    __LOG_PREFIX__ = "NpMesh()"

    def __init__(self, mesh: DefaultDict[str, list]) -> None:
        """
            Initialize the mesh object.
            Input parameters:
                - mesh: dictionary of objects associated with the 3D file
        """
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
        return utils.convert_to_numpy_matrix(self.raw_mesh['v'])

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
        return utils.convert_to_numpy_matrix(self.raw_mesh['vt'])
    
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
        return utils.convert_to_numpy_matrix(self.raw_mesh['vn'])
    
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
        return utils.convert_to_numpy_matrix(self.raw_mesh['f'])
    
    def _get_face_vertices(self) -> np.ndarray:
        """
            Get the face vertices from the mesh.
            Output:
                - numpy array of face vertices
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face vertices from the mesh")
        return self.faces_combo[:, :, 0].astype(int)
    
    def _get_face_texture_coordinates(self) -> np.ndarray:
        """
            Get the face texture coordinates from the mesh.
            Output:
                - numpy array of face texture coordinates
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face texture coordinates from the mesh")
        return self.faces_combo[:, :, 1].astype(int)
    
    def _get_face_vertex_normals(self) -> np.ndarray:
        """
            Get the face vertex normals from the mesh.
            Output:
                - numpy array of face vertex normals
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face vertex normals from the mesh")
        return self.faces_combo[:, :, 2].astype(int)

    def _set_face_items(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Get the face items from the mesh.
            Output:
                - numpy array of face items
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the face items of the mesh")
        return self._get_face_vertices(), self._get_face_texture_coordinates(), self._get_face_vertex_normals()
    
    def _build_mesh(self) -> None:
        """
            Build the mesh from the raw mesh data.
        """
        try:
            logger.info(f"{self.__LOG_PREFIX__}: Building the mesh from the raw mesh data")
            self.vertices = self._get_vertices()
            self.texture_coordinates = self._get_texture_coordinates()
            self.vertex_normals = self._get_vertex_normals()
            self.faces_combo = self._get_faces()
            self.faces, self.faces_texture_coordinates, self.faces_texture_coordinates = self._set_face_items()
            self.remesh = Remesh(self.vertices, self.faces)
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while building the mesh")
            raise e
        
    def loop_subdivision(self, iterations=1) -> Tuple[np.ndarray, np.ndarray]:
        """
            Apply Loop subdivision to the input mesh for the specified number of iterations.
            Input parameters:
                - iterations: number of iterations
            Output:
                - vertices and faces after subdivision
        """
        try:
            logger.info(f"{self.__LOG_PREFIX__}: Applying Loop subdivision to the input mesh")
            new_vertices, new_faces = self.remesh.loop_subdivision(iterations=iterations)
            return new_vertices, new_faces
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while applying Loop subdivision to the input mesh")
            raise e
    
    @staticmethod
    def save(vertices: np.ndarray, faces: np.ndarray, file_path: str) -> None:
        """
            Save the vertices and faces to a .obj file.
            Input parameters:
                - vertices: numpy array of vertices
                - faces: numpy array of faces
                - file_path: path to the file
        """
        try:
            if file_path is None or ".obj" not in file_path:
                logger.error(f"{NpMesh.__LOG_PREFIX__}: Invalid file path")
                raise ValueError("Invalid file path")
            logger.info(f"{NpMesh.__LOG_PREFIX__}: Saving the data in the file: {file_path}")
            utils.save_obj(vertices, faces, file_path)
        except Exception as e:
            logger.error(f"{NpMesh.__LOG_PREFIX__}: Error while saving the data in the file: {file_path}")
            raise e