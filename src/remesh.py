##########################################################################################################
# Any remeshing angorithms should be implemented here.
# Algorithms implemented:
#   - Loop Subdivision
##########################################################################################################

import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Callable
import mpl_toolkits.mplot3d as a3
from src.utils.utils import get_edges, validate_triangulated_mesh, get_matching_row_indices

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
    def _plot(p_vertices: np.ndarray, p_edges: np.ndarray, c_vertices: np.ndarray, c_edges: np.ndarray = None, figsize: Tuple[int, int] = (10, 7), p_alpha: float = 0.25, x_label: str = 'X', y_label: str = 'Y', z_label: str= 'Z') -> None:
        """
        Plot the vertices of the mesh.
        Input parameters:
            - p_vertices: numpy ndarray of parent/og vertices
            - p_edges: numpy ndarray of parent/og edges
            - c_vertices: numpy ndarray of child vertices
            - c_edges: numpy ndarray of child edges
            - figsize: figure size
            - p_alpha: alpha value for the parent/og edges
            - x_label: x-axis label
            - y_label: y-axis label
            - z_label: z-axis label
        Output: None
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Plotting the vertices of the mesh")
        ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
        v_p_edges = Remesh._get_vertices_from_edges(p_vertices, p_edges)
        if c_edges is not None:
            v_c_edges = Remesh._get_vertices_from_edges(c_vertices, c_edges)
        ax.scatter(p_vertices[:, 0], p_vertices[:, 1], p_vertices[:, 2], label='Parent Vertices', c='r')
        ax.scatter(c_vertices[:, 0], c_vertices[:, 1], c_vertices[:, 2], label='Child Vertices', c='orange')
        for edge in v_p_edges:
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c='r', alpha=p_alpha)
        if c_edges is not None:
            for edge in v_c_edges:
                ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c='orange')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.show()
    
    @staticmethod
    def _plot_faces(vertex: np.ndarray, faces: np.ndarray, figsize: Tuple[int, int] = (10, 7), alpha: float = 0.5, x_label: str = 'X', y_label: str = 'Y', z_label: str = 'Z') -> None:
        """
        Plot the faces of the mesh.
        Input parameters:
            - vertex: numpy ndarray of vertices, (n x 3)
            - faces: numpy ndarray of faces, (m x 3) - one indexed
            - figsize: figure size
            - alpha: alpha value for the faces
            - x_label: x-axis label
            - y_label: y-axis label
            - z_label: z-axis label
        Output: None
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Plotting the faces of the mesh")
        if faces.shape[-1] != 3:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid faces")
            raise ValueError("Invalid faces")
        ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
        for face in faces:
            vertices = Remesh._get_vertices_from_edges(vertex, face)
            poly = a3.art3d.Poly3DCollection([vertices], alpha=alpha)
            poly.set_color(np.random.rand(3,))
            ax.add_collection3d(poly)
            ax.plot(vertex[face[0]-1, 0], vertex[face[1]-1, 1], vertex[face[2]-1, 2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.show()

    @staticmethod
    def _get_vertices_from_edges(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Get the vertices from the edges of the mesh.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of edges, one indexed (it can be pseduo-edge as well)
        Output:
            - numpy ndarray of vertices (m x 2 x 3)
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the vertices from the edges of the mesh")
        return vertices[edges - 1]
    
    @staticmethod
    def _get_vertex_face_mask(faces: np.ndarray, edges: np.ndarray, squeeze: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the vertex-face mask for the given edges of the mesh.
        i.e. Say you a set of faces, the vertex-face mask is a boolean mask that represents
            1. edge[:, 0] is in face
            2. edge[:, 1] is in face
        Input parameters:
            - faces: numpy ndarray of faces, (n x 3) - one indexed
            - edges: numpy ndarray of edges, (m x 2) - one indexed
            - squeeze: boolean flag to squeeze the resultant mask, default is True
        Output:
            - a tuple of numpy ndarrays representing the vertex-face mask
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Getting the vertex-face mask")
        # For supporting numpy broadcasting, we need to expand the dimensions of the faces and edges
        # 1. Expand the dimensions of the faces
        faces_expanded = faces[:, np.newaxis, :] # (n, 1, 3)
        # 2. Expand the dimensions of the edges
        edges_expanded = edges[np.newaxis, :, :] # (1, m, 2)
        # 3. Compare each face with each edge, checking for the presence of both vertices of an edge in a face
        if squeeze:
            v1 = np.any(faces_expanded == edges_expanded[..., 0, np.newaxis], axis=2)
            v2 = np.any(faces_expanded == edges_expanded[..., 1, np.newaxis], axis=2)
        else:
            v1 = faces_expanded == edges_expanded[..., 0, np.newaxis]
            v2 = faces_expanded == edges_expanded[..., 1, np.newaxis]
        return v1, v2

    @staticmethod
    def _get_face_mask(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Get the face mask for the given edges of the mesh, in other words
        get all the faces that contains atleast one edge from the given set of edges.
        Input parameters:
            - faces: numpy ndarray of faces, (n x 3) - one indexed
            - edges: numpy ndarray of edges, (m x 2) - one indexed
        Output:
            -  numpy ndarray of face mask
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Getting the face mask for the given set of edges of the mesh")
        # 1. Get the vertex-face mask
        v1, v2 = Remesh._get_vertex_face_mask(faces, edges)
        # 2. A face contains an edge if it contains both vertices of the edge, get the mask
        mask = v1 & v2
        # 3. Identify faces that contains atleast one edge
        face_mask = np.any(mask, axis=1)
        return face_mask
    
    @staticmethod
    def _get_opposite_face_vertex_indices(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Get the vertices of the opposite faces for the given set of edges of the mesh.
        Input parameters:
            - faces: filtered faces, numpy ndarray of faces, (n x 3) - one indexed
            - edges: numpy ndarray of edges, (m x 2) - one indexed
        Output:
            - numpy ndarray of opposite face vertices
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Getting the vertices of the opposite faces for the given set of edges of the mesh")
        # 1. Get the vertex-face mask
        v1, v2 = Remesh._get_vertex_face_mask(faces, edges, squeeze=False)
        # 2. Get face mask that represents the presence of either the first or the second vertex of the edge
        mask = v1 | v2
        # 3. Get the sum of the mask - represents the number of vertiecs that forms an edge in a face
        mask_sum = np.sum(mask, axis=-1)
        # 4. Mark all the position that has a value 2 - 2 means both vertices of the edge are use to form a face boundary
        # Transpose the resultant for ease of use
        # import pdb; pdb.set_trace()
        mask_sum = (mask_sum == 2).T
        # 5. Check for correctness - only one interior edge is shared by two faces
        if not np.all(mask_sum.sum(axis=-1) == 2):
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid mask sum while getting the opposite face vertices")
            raise ValueError("Invalid mask sum while getting the opposite face vertices") 
        # 6. Get the faces that an interior edge belongs to
        _, face_indices = np.where(mask_sum)
        foi = faces[face_indices].reshape(len(edges), -1, 3) # (m, 2, 3)
        # 7. Get the vertices of the opposite faces
        e1, e2 = edges, edges[:, ::-1]
        e1_expanded, e2_expanded = e1[:, :, None], e2[:, :, None]
        # (m, 2, 3) -> axis = 2 - only one item will be non-zero due to the nature of the operation and representation
        filtered_vertices = foi * (foi != e1_expanded) * (foi != e2_expanded)
        result = filtered_vertices[filtered_vertices != 0].reshape(-1, 2) # (m, 2)
        return result
    
    @staticmethod
    def _get_odd_vertices(vertices: np.ndarray, edges: np.ndarray, interior_edges: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Get the odd vertices of the mesh.
        1. Interior edges
             c
          /     \\
        a - o_v - b
         \\     /
             d
        2. Boundary edges
        a - o_v - b

        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of unique edges (p x 2) - one indexed
            - unique_edges: numpy ndarray of unique edges (x x 2) - one indexed
            - interior_edges: numpy ndarray of interior edges (y, 2) - one indexed
            - boundary_edges: numpy ndarray of boundary edges (z, 2) - one indexed
            - faces: numpy ndarray of faces (n x 3) - one indexed
        Output:
            - numpy ndarray of odd vertices
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Getting the odd vertices of the mesh")
        # 1. Set the odd vertices of the mesh to the middle of the edges - default/boundary cases
        odd_vertices = Remesh._get_vertices_from_edges(vertices, edges).mean(axis=1)
        # 2. Set the odd vertices of the mesh - interior edges
        # 2.a. Get the vertices (a, b) of the interior edges
        e_ab = Remesh._get_vertices_from_edges(vertices, interior_edges)
        e_a, e_b = e_ab[:, 0, :], e_ab[:, 1, :]
        # 2.b. Get vertices (c, d) form their opposite faces
        # 2.b.1. Get face mask, containing all the interior edges
        face_mask = Remesh._get_face_mask(faces, interior_edges)
        # 2.b.2. Get face of interest
        foi = faces[face_mask]
        # 2.b.3. Get the vertices of the opposite faces
        e_cd = Remesh._get_opposite_face_vertex_indices(foi, interior_edges)
        e_cd = Remesh._get_vertices_from_edges(vertices, e_cd)
        e_c, e_d = e_cd[:, 0, :], e_cd[:, 1, :]
        return odd_vertices

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
            odd_vertices = Remesh._get_odd_vertices(vertices, edges, interior_edges, faces)
            if Remesh.__VERBOSE__:
                # Remesh._plot(vertices, edges, odd_vertices)
                # Remesh._plot_faces(vertices, faces)
                pass
            break
            # Check if edges are shared by more than 2 faces
            # Correct edge orientation if needed
            # Calculate odd edges
            # Calculate even vertices
            # Handle boundary vertices
            # Calculate new vertices
            # Calculate new faces
        return vertices, faces