##########################################################################################################
# Any remeshing angorithms should be implemented here.
# Algorithms implemented:
#   - Loop Subdivision
##########################################################################################################

import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
from src.utils.utils import get_edges, validate_triangulated_mesh

logger = logging.getLogger(__name__)


class Remesh:
    
    """
        Class to represent remeshing algorithms.
    """

    __LOG_PREFIX__ = "Remesh()"
    __VERBOSE__ = True

    def __init__(self, vertices: np.ndarray = None, faces: np.ndarray = None) -> None:
        """
        Initialize the remesh object.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, note that the faces should be triangulated
        """
        if vertices is None or faces is None:
            logger.error(f"{self.__LOG_PREFIX__}: Invalid vertices or faces")
            raise ValueError("Invalid vertices or faces")
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the remesh object")
        self.vertices = vertices
        self.faces = faces

    def loop_subdivision(self, iterations=1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Loop subdivision to the input mesh for the specified number of iterations.
        Loop subdivision:
            - For triangulated meshes
        Loop subdivision is is achieved by splitting each triangle into four smaller triangles 
        and apprxoimating thier positions to get a smoother surface.
        Input parameters:
            - iterations: number of iterations
        Output:
            - vertices and faces after subdivision
        """
        logger.info(f"{self.__LOG_PREFIX__}: Applying Loop subdivision to the input mesh")
        new_vertices, new_faces = self._loop_subdivision(vertices=self.vertices, faces=self.faces, iterations=iterations)
        logger.info(f"{self.__LOG_PREFIX__}: Loop subdivision applied successfully")
        return new_vertices, new_faces
    
    @staticmethod
    def _get_vertices_from_edges(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Get the vertices from the edges of the mesh.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of edges, one indexed
        Output:
            - numpy ndarray of vertices (m x 2 x 3)
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Getting the vertices from the edges of the mesh")
        return vertices[edges - 1]
    
    @staticmethod
    def _plot(p_vertices: np.ndarray, p_edges: np.ndarray, c_vertices: np.ndarray, c_edges: np.ndarray = None) -> None:
        """
        Plot the vertices of the mesh.
        Input parameters:
            - p_vertices: numpy ndarray of parent/og vertices
            - p_edges: numpy ndarray of parent/og edges
            - c_vertices: numpy ndarray of child vertices
            - c_edges: numpy ndarray of child edges
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Plotting the vertices of the mesh")
        ax = plt.figure().add_subplot(projection='3d')
        v_p_edges = Remesh._get_vertices_from_edges(p_vertices, p_edges).reshape(-1, 3)
        if c_edges is not None:
            v_c_edges = Remesh._get_vertices_from_edges(c_vertices, c_edges).reshape(-1, 3)
        ax.scatter(p_vertices[:, 0], p_vertices[:, 1], p_vertices[:, 2], label='Parent Vertices', c='r')
        ax.scatter(c_vertices[:, 0], c_vertices[:, 1], c_vertices[:, 2], label='Child Vertices', c='orange')
        ax.plot(v_p_edges[:, 0], v_p_edges[:, 1], v_p_edges[:, 2], label='Parent Edges', c='r', alpha=0.25)
        if c_edges is not None:
            ax.plot(v_c_edges[:, 0], v_c_edges[:, 1],
                    v_c_edges[:, 2], label='Child Edges', c='orange')
        plt.show()
    
    @staticmethod
    def _get_odd_vertices(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Get the odd vertices of the mesh.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of unique edges (p x 2) - one indexed
        Output:
            - numpy ndarray of odd vertices
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Getting the odd vertices of the mesh")
        return Remesh._get_vertices_from_edges(vertices, edges).mean(axis=1)

    @staticmethod
    def _loop_subdivision(vertices: np.ndarray, faces: np.ndarray, iterations: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Driver code for implementing Loop Subdivision: apply Loop subdivision to the input mesh.
        The algorithm is implemented using matrix operations to improve performance by avoiding loops.
        Algorithm:
            TODO: Add the algorithm here
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, note that the faces should be triangulated
            - iterations: number of iterations, default is 1
        Output:
            - vertices and faces after subdivision
        """
        if vertices is None or faces is None:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid vertices or faces")
            raise ValueError("Invalid vertices or faces")
        if iterations is None or iterations < 1:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid iteration parameter")
            raise ValueError("Invalid iteration parameter")
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Applying Loop subdivision algorithm to the input mesh")
        # For each iteration, apply the subdivision algorithm
        for _ in tqdm(range(iterations)):
            # Get all the edges of the mesh
            edges, face_indices, \
                unique_edges, unique_edge_indices, unique_inverse, \
                    interior_edges, boundary_edges, \
                        left_interior_edge_matches, right_interior_edge_matches, \
                            left_boundary_edge_matches, right_boundary_edge_matches = get_edges(faces)
            # Validate that any edge is not shared by more than 2 faces
            if not validate_triangulated_mesh(edges):
                logger.error(f"{Remesh.__LOG_PREFIX__}: One or more edges are shared by more than 2 faces")
                raise ValueError("One or more edges are shared by more than 2 faces")
            # Get the odd vertices of the mesh
            odd_vertices = Remesh._get_odd_vertices(vertices, edges)
            if Remesh.__VERBOSE__:
                Remesh._plot(vertices, edges, odd_vertices)
            break
            # Check if edges are shared by more than 2 faces
            # Correct edge orientation if needed
            # Calculate odd edges
            # Calculate even vertices
            # Handle boundary vertices
            # Calculate new vertices
            # Calculate new faces
        return vertices, faces