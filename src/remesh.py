##########################################################################################################
# Any remeshing angorithms should be implemented here.
# Algorithms implemented:
#   - Loop Subdivision
#   - Decimation using Quadric Error Metrics
##########################################################################################################

import heapq
import logging
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from src.utils.utils import (
    get_edges_from_faces_vstack, get_edges_from_faces_hstack, get_special_edges,
    validate_triangulated_mesh, get_matching_row_indices, get_incident_matrix
)


logger = logging.getLogger(__name__)


class Remesh:

    """
        Class to represent remeshing algorithms.
    """

    __LOG_PREFIX__ = "Remesh()"
    __PLOT_EVERY__ = 1
    __EPSILON__ = 1e-13
    __PLOT_DIRECTORY__ = "../assets/plot/assignment1"

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

    def loop_subdivision(self, iterations: int = 1, fixed: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Loop subdivision to the input mesh for the specified number of iterations.
        Loop subdivision:
            - For triangulated meshes
        Loop subdivision is is achieved by splitting each triangle into four smaller triangles 
        and apprxoimating thier positions to get a smoother surface.
        Input parameters:
            - iterations: number of iterations
            - fixed: boolean indicating whether to keep the original vertices of the mesh fixed during mesh operations
            - verbose: boolean indicating whether to make verbose
        Output:
            - vertices and faces after subdivision
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Applying Loop subdivision to the input mesh")
        new_vertices, new_faces = self._loop_subdivision(
            vertices=self.vertices, faces=self.faces, iterations=iterations, fixed=fixed, verbose=verbose)
        logger.info(
            f"{self.__LOG_PREFIX__}: Loop subdivision applied successfully")
        return new_vertices, new_faces

    def decimation(self, faces: int = 3, fixed: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply decimation to the input mesh for the specified number of faces.
        Decimation:
            - For triangulated meshes
            - Done using Quadric Error Metrics
        Decimation is the process of reducing the number of faces in a mesh.
        Input parameters:
            - faces: number of faces
            - fixed: boolean indicating whether to keep the original vertices of the mesh fixed during mesh operations
            - verbose: boolean indicating whether to make verbose
        Output:
            - vertices and faces after decimation
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Applying decimation to the input mesh")
        new_vertices, new_faces = self._decimation(
            vertices=self.vertices, faces=self.faces, reduce_faces_to=faces, fixed=fixed, verbose=verbose)
        logger.info(f"{self.__LOG_PREFIX__}: Decimation applied successfully")
        return new_vertices, new_faces

    @staticmethod
    def _plot(p_vertices: np.ndarray, p_edges: np.ndarray, c_vertices: np.ndarray, c_edges: np.ndarray = None, figsize: Tuple[int, int] = (12, 8), p_alpha: float = 0.25, x_label: str = 'X', y_label: str = 'Y', z_label: str = 'Z', text_offset: float = 0.1, title: str = "3D plot") -> None:
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
            - text_offset: text offset for the vertices
            - title: title of the plot
        Output: None
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Plotting the vertices of the mesh")
        ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
        v_p_edges = Remesh._get_vertices_from_idx_or_edges(p_vertices, p_edges)
        if c_edges is not None:
            v_c_edges = Remesh._get_vertices_from_idx_or_edges(
                c_vertices, c_edges)
        ax.scatter(p_vertices[:, 0], p_vertices[:, 1],
                   p_vertices[:, 2], label='Parent Vertices', c='r')
        for idx, edge in enumerate(p_vertices):
            ax.text(edge[0]+text_offset, edge[1] +
                    text_offset, edge[2]+text_offset, c='r', s=f"{idx+1}")
        ax.scatter(c_vertices[:, 0], c_vertices[:, 1],
                   c_vertices[:, 2], label='Child Vertices', c='orange')
        for idx, edge in enumerate(c_vertices):
            ax.text(edge[0]+text_offset, edge[1] +
                    text_offset, edge[2]+text_offset, c='orange', s=f"{idx+1}")
        for edge in v_p_edges:
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c='r', alpha=p_alpha)
        if c_edges is not None:
            for edge in v_c_edges:
                ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], c='orange')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        text = f"Parent Vertices: {p_vertices.shape}\nParent Edges: {p_edges.shape}\nChild Vertices: {c_vertices.shape if c_vertices is not None else 'N/A'}\nChild Edges: {c_edges.shape if c_edges is not None else 'N/A'}"
        ax.text2D(0.0, 1.05, text, transform=ax.transAxes, ha='left', va='top')
        ax.text2D(0.5, 1.1, title, transform=ax.transAxes,
                  ha='center', va='top')
        ax.legend(loc='upper right')
        plt.show()

    @staticmethod
    def _plot_faces(vertex: np.ndarray, faces: np.ndarray, figsize: Tuple[int, int] = (12, 8), alpha: float = 0.5, x_label: str = 'X', y_label: str = 'Y', z_label: str = 'Z', title: str = "3D face plot", save: bool = False, file_name: str = "face") -> None:
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
            - title: title of the plot
        Output: None
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Plotting the faces of the mesh")
        if faces.shape[-1] != 3:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid faces")
            raise ValueError("Invalid faces")
        ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
        for face in faces:
            vertices = Remesh._get_vertices_from_idx_or_edges(vertex, face)
            poly = a3.art3d.Poly3DCollection([vertices], alpha=alpha)
            poly.set_color(np.random.rand(3,))
            ax.add_collection3d(poly)
            ax.plot(vertex[face[0]-1, 0],
                    vertex[face[1]-1, 1], vertex[face[2]-1, 2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        ax.text2D(0.5, 1.0, title, transform=ax.transAxes,
                  ha='center', va='top')
        if save:
            plt.savefig(f"{Remesh.__PLOT_DIRECTORY__}/{file_name}.png")
        plt.show()

    @staticmethod
    def _get_edges(faces: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Get all the edges and a set of unique edges of the mesh given a list of faces.
        Input parameters:
            - faces: numpy ndarray of faces (n x 3)
        Output:
            - og_edges: numpy ndarray of original edges (m x 2)
            - edges: numpy ndarray of (sorted) edges (m x 2)
            - face_indices: numpy ndarray of face indices (m,)
            - unique_edges: numpy ndarray of unique edges (p x 2)
            - unique_edge_indices: numpy ndarray of unique edge indices (p,)
            - unique_inverse: numpy ndarray of unique inverse (m,)
            - interior_edges: numpy ndarray of interior edges (x, 2)
            - boundary_edges: numpy ndarray of boundary edges (y, 2)
            - left_interior_edge_matches: numpy ndarray of left interior edge matches (z,)
            - right_interior_edge_matches: numpy ndarray of right interior edge matches (z,)
            - left_boundary_edge_matches: numpy ndarray of left boundary edge matches (w,)
            - right_boundary_edge_matches: numpy ndarray of right boundary edge matches (w,)
        """
        edges, og_edges = get_edges_from_faces_vstack(
            faces), get_edges_from_faces_hstack(faces)
        edges.sort(axis=1)
        unique_edges, unique_edge_indices, unique_inverse = np.unique(
            edges, axis=0, return_index=True, return_inverse=True)
        face_indices = np.tile(np.arange(len(faces)), 3)
        interior_edges, boundary_edges = get_special_edges(
            unique_edges, unique_inverse)
        left_interior_edge_matches, right_interior_edge_matches = get_matching_row_indices(
            edges, interior_edges)
        left_boundary_edge_matches, right_boundary_edge_matches = get_matching_row_indices(
            edges, boundary_edges)
        return og_edges, edges, face_indices, \
            unique_edges, unique_edge_indices, unique_inverse, \
            interior_edges, boundary_edges, \
            left_interior_edge_matches, right_interior_edge_matches, \
            left_boundary_edge_matches, right_boundary_edge_matches

    @staticmethod
    def _get_vertices_from_idx_or_edges(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Get the vertices from the idx of vertex or edges of the mesh.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of edges, one indexed (it can be pseduo-edge as well)
                    Even granular, it can ve simply an array of indices representing a side of the edges
        Output:
            - numpy ndarray of vertices (m x 2 x 3) | (m x 3)
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the vertices from the edges of the mesh")
        return vertices[edges - 1]

    @staticmethod
    def _get_vertex_face_mask(faces: np.ndarray, edges: np.ndarray, squeeze: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the vertex-face mask for the given edges of the mesh.
        i.e. Say you have a set of faces, the vertex-face mask is a boolean mask that represents
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
        faces_expanded = faces[:, np.newaxis, :]  # (n, 1, 3)
        # 2. Expand the dimensions of the edges
        edges_expanded = edges[np.newaxis, :, :]  # (1, m, 2)
        # 3. Compare each face with each edge, checking for the presence of both vertices of an edge in a face
        if squeeze:
            v1 = np.any(faces_expanded ==
                        edges_expanded[..., 0, np.newaxis], axis=2)
            v2 = np.any(faces_expanded ==
                        edges_expanded[..., 1, np.newaxis], axis=2)
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
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the face mask for the given set of edges of the mesh")
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
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the vertices of the opposite faces for the given set of edges of the mesh")
        # 1. Get the vertex-face mask
        v1, v2 = Remesh._get_vertex_face_mask(faces, edges, squeeze=False)
        # 2. Get face mask that represents the presence of either the first or the second vertex of the edge
        mask = v1 | v2
        # 3. Get the sum of the mask - represents the number of vertiecs that forms an edge in a face
        mask_sum = np.sum(mask, axis=-1)
        # 4. Mark all the position that has a value 2 - 2 means both vertices of the edge are use to form a face boundary
        # Transpose the resultant for ease of use
        mask_sum = (mask_sum == 2).T
        # 5. Check for correctness - only one interior edge is shared by two faces
        if not np.all(mask_sum.sum(axis=-1) == 2):
            logger.error(
                f"{Remesh.__LOG_PREFIX__}: Invalid mask sum while getting the opposite face vertices")
            raise ValueError(
                "Invalid mask sum while getting the opposite face vertices")
        # 6. Get the faces that an interior edge belongs to
        _, face_indices = np.where(mask_sum)
        foi = faces[face_indices].reshape(len(edges), -1, 3)  # (m, 2, 3)
        # 7. Get the vertices of the opposite faces
        e1, e2 = edges, edges[:, ::-1]
        e1_expanded, e2_expanded = e1[:, :, None], e2[:, :, None]
        # (m, 2, 3) -> axis = 2 - only one item will be non-zero due to the nature of the operation and representation
        filtered_vertices = foi * (foi != e1_expanded) * (foi != e2_expanded)
        result = filtered_vertices[filtered_vertices !=
                                   0].reshape(-1, 2)  # (m, 2)
        return result

    @staticmethod
    def _get_odd_vertices(vertices: np.ndarray, edges: np.ndarray, interior_edges: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Get the odd vertices of the mesh.
        1. Interior vertices
             c
          /     \\
        a - o_v - b
         \\     /
             d
        2. Boundary vertices
        a - o_v - b

        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of unique edges (p x 2) - one indexed
            - interior_edges: numpy ndarray of interior edges (x, 2) - one indexed
            - faces: numpy ndarray of faces (n x 3) - one indexed
        Output:
            - numpy ndarray of odd vertices
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the odd vertices of the mesh")
        # 0. Get vertex matches between the edges and the interior edges
        left_interior_edge_matches, right_interior_edge_matches = get_matching_row_indices(
            edges, interior_edges)
        # 1. Set the odd vertices of the mesh to the middle of the edges - default/boundary cases
        odd_vertices = Remesh._get_vertices_from_idx_or_edges(
            vertices, edges).mean(axis=1)
        # 2. Set the odd vertices of the mesh - interior edges
        # 2.a. Get the vertices (a, b) of the interior edges
        # Formula: Boundary vertices - 1.0/2.0 * (a + b)
        e_ab = Remesh._get_vertices_from_idx_or_edges(vertices, interior_edges)
        # rearrange the vertices as per the matches
        e_ab = e_ab[right_interior_edge_matches]
        e_a, e_b = e_ab[:, 0, :], e_ab[:, 1, :]
        # 2.b. Get vertices (c, d) form their opposite faces
        # 2.b.1. Get face mask, containing all the interior edges
        face_mask = Remesh._get_face_mask(faces, interior_edges)
        # 2.b.2. Get face of interest
        foi = faces[face_mask]
        # 2.b.3. Get the vertices of the opposite faces
        e_cd_idx = Remesh._get_opposite_face_vertex_indices(
            foi, interior_edges)
        # rearrange the vertices as per the matches
        e_cd_idx = e_cd_idx[right_interior_edge_matches]
        e_cd = Remesh._get_vertices_from_idx_or_edges(vertices, e_cd_idx)
        e_c, e_d = e_cd[:, 0, :], e_cd[:, 1, :]
        # Formula: Interior vertices - 3/8 * (a + b) + 1/8 * (c + d)
        odd_vertices[left_interior_edge_matches] = (
            3/8) * (e_a + e_b) + (1/8) * (e_c + e_d)
        return odd_vertices

    @staticmethod
    def _get_even_vertices(vertices: np.ndarray, edges: np.ndarray, interior_edges: np.ndarray, boundary_edges: np.ndarray) -> np.ndarray:
        """
        Get the even vertices of the mesh.
        1. Interior vertices
            Imagine a closed polygon with n vertices, similar to a hexagon and its features, the center of the polygon is the even vertex, eg:
                 a - b
               /  \\/  \\
               c - e_v - d
                \\ /\\  /
                  e - f
        2. Boundary vertices
        a - e_v - b

        Input parameters:
            - vertices: numpy ndarray of vertices
            - edges: numpy ndarray of unique edges (m x 2) - one indexed
            - interior_edges: numpy ndarray of interior edges (x, 2) - one indexed
            - boundary_edges: numpy ndarray of boundary edges (y, 2) - one indexed
        Output:
            - numpy ndarray of even vertices
        """
        # 1. Get neighbors of the vertices
        incident_matrix, vertices_idx, k = get_incident_matrix(edges)
        # 2. Calculate beta
        # Formula: beta = (1/k) * (5/8 - (3/8 + 1/4 * cos(2 * pi / k)) ** 2)
        k = np.maximum(k, Remesh.__EPSILON__) # to avoid division by zero, just in case
        beta = (1/k) * ((5/8) - ((3/8) + (1/4) * np.cos(2 * np.pi / k)) ** 2)
        # 3. Get neighbors of the vertices
        k_neighbors = incident_matrix * vertices_idx
        k_neighbor_raw_vertices = Remesh._get_vertices_from_idx_or_edges(
            vertices, k_neighbors)
        k_neighbor_vertices = k_neighbor_raw_vertices * \
            incident_matrix[:, :, None]
        # 4. Calculate the even vertices
        # Formula: v = (1 - k * beta) * v + beta * (sum of all k neighbors vertices)
        even_vertices = Remesh._get_vertices_from_idx_or_edges(
            vertices, vertices_idx)
        even_vertices = even_vertices * (1 - k[:, None] * beta[:, None]) + \
            beta[:, None] * (k_neighbor_vertices.sum(axis=1))
        # 5. Handle boundary vertices
        # Formula: v = (1/8) * (a + b) + (3/4) * v
        boundary_vertices = np.unique(boundary_edges)
        # -1 for zero indexing w.r.t k_neighbor_vertices
        boundary_k_neighbor_vertices = k_neighbor_vertices[boundary_vertices-1, :, :]
        boundary_even_vertices = Remesh._get_vertices_from_idx_or_edges(
            vertices, boundary_vertices)
        even_boundary_vertices = (1/8) * boundary_k_neighbor_vertices.sum(axis=1) + (
            3/4) * boundary_even_vertices  # rearrange to make sure vertices as per the matches
        even_vertices[boundary_vertices-1] = even_boundary_vertices
        return even_vertices

    @staticmethod
    def _update_faces(faces: np.ndarray, edges: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
        """
        Update the faces of the mesh.
        Input parameters:
            - faces: numpy ndarray of original faces
            - edges: numpy ndarray of original edges (hstacked), one indexed
            - unique_edges: numpy ndarray of unique edges (p x 2) - one indexed
        Output:
            - numpy ndarray of updated faces
        """
        logger.info(f"{Remesh.__LOG_PREFIX__}: Updating the faces of the mesh")
        # 1. Conver the hstacked OG edges from (f x 6) to (f x 3 x 2)
        edges = edges.reshape(-1, 3, 2)
        # 2. Sort the edges here to smooth the process (easier representation)
        edges.sort(axis=-1)
        # 3. Flatten the edges
        edges = edges.reshape(-1, 2)
        # 4. Get the indices of the unique edges into edges
        matches = np.all(edges[:, None] == unique_edges, axis=-1)
        # Get on the second axis - second axis represents the unique edges
        odd_points = np.where(matches)[1]
        # 5. Reshape the odd_points to (f x 3)
        odd_points = odd_points.reshape(-1, 3)
        # 6. Update odd_points to start indexing from the total number of old vertices (or even vertices)
        odd_points += len(np.unique(unique_edges))
        # 7. Standardize the faces by ensuring that the face vertices starts from 1 for odd points by adding 1.
        odd_points += 1
        # 8. Construct new face
        new_face = np.column_stack([
            faces[:, 0],  # T1
            odd_points[:, 0],  # T1
            odd_points[:, 2],  # T1
            odd_points[:, 0],  # T2
            faces[:, 1],  # T2
            odd_points[:, 1],  # T2
            odd_points[:, 2],  # T3
            odd_points[:, 1],  # T3
            faces[:, 2],  # T3
            odd_points[:, 0],  # T4
            odd_points[:, 1],  # T4
            odd_points[:, 2]  # T4
        ])
        # 9. Resize the new face to (f x 3)
        new_face = new_face.reshape(-1, 3)
        return new_face

    @staticmethod
    def _loop_subdivision(vertices: np.ndarray, faces: np.ndarray, iterations: int = 1, fixed: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Driver code for implementing Loop Subdivision: apply Loop subdivision to the input mesh.
        The algorithm is implemented using matrix operations to improve performance by avoiding loops.
        Algorithm:
            1. For each iteration, apply the subdivision
                a. Get all the edges of the mesh
                b. Get the odd vertices of the mesh
                c. Get the even vertices of the mesh
                d. Update the faces and vertices of the mesh
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, note that the faces should be triangulated
            - iterations: number of iterations, default is 1
            - fixed: boolean indicating whether to keep the original vertices of the mesh fixed during mesh operations
            - verbose: boolean indicating whether to make verbose
        Output:
            - vertices and faces after subdivision
        """
        if vertices is None or faces is None:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid vertices or faces")
            raise ValueError("Invalid vertices or faces")
        if iterations is None or iterations < 1:
            logger.error(
                f"{Remesh.__LOG_PREFIX__}: Invalid iteration parameter")
            raise ValueError("Invalid iteration parameter")
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Applying Loop subdivision algorithm to the input mesh")
        # For each iteration, apply the subdivision algorithm
        for run in range(iterations):
            logger.info(
                f"{Remesh.__LOG_PREFIX__}: Applying Loop subdivision algorithm to the input mesh: Iteration {run+1} | {len(vertices)} Vertices | {len(faces)} Faces")
            # Get all the edges of the mesh
            og_edges, edges, face_indices, \
                unique_edges, unique_edge_indices, unique_inverse, \
                interior_edges, boundary_edges, \
                left_interior_edge_matches, right_interior_edge_matches, \
                left_boundary_edge_matches, right_boundary_edge_matches = Remesh._get_edges(
                    faces)
            # Validate that any edge is not shared by more than 2 faces
            if not validate_triangulated_mesh(edges):
                logger.error(
                    f"{Remesh.__LOG_PREFIX__}: One or more edges are shared by more than 2 faces")
                raise ValueError(
                    "One or more edges are shared by more than 2 faces")
            # Get the odd vertices of the mesh
            odd_vertices = Remesh._get_odd_vertices(
                vertices, unique_edges, interior_edges, faces)
            # Get the even vertices of the mesh
            even_vertices = Remesh._get_even_vertices(
                vertices, unique_edges, interior_edges, boundary_edges)
            # Update the faces of the mesh
            new_faces = Remesh._update_faces(faces, og_edges, unique_edges)
            # Merge the odd and even vertices
            # Note the order - required for the face generation
            new_vertices = np.vstack((even_vertices, odd_vertices))
            # Plot for visualization
            if verbose and run % Remesh.__PLOT_EVERY__ == 0:
                Remesh._plot(vertices, unique_edges, odd_vertices,
                             title=f"Loop Subdivision: Iteration {run+1} - Odd Vertices")
                Remesh._plot(vertices, unique_edges, even_vertices,
                             title=f"Loop Subdivision: Iteration {run+1} - Even Vertices")
                Remesh._plot(vertices, unique_edges, new_vertices,
                             title=f"Loop Subdivision: Iteration {run+1} - Merged Vertices | {len(new_vertices)} Vertices")
                Remesh._plot_faces(
                    vertices, faces, title=f"3D Plot - Original Faces at the start of this iteration | {len(faces)} Faces")
                Remesh._plot_faces(
                    new_vertices, new_faces, title=f"3D Plot - Loop Subdivision: Iteration {run+1} - New Faces | {len(new_faces)} Faces", save=True, file_name=f"loop_subdivision_{run+1}")
            # update the vertices and faces
            vertices, faces = new_vertices, new_faces
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Loop subdivision algorithm resulted in: {len(vertices)} Vertices | {len(faces)} Faces")
        return vertices, faces

    @staticmethod
    def _get_face_normals_and_d(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Get the face normals of the mesh and the constand 'd'.
        Plane: 
            Q ---- P
            |
            |
            R
        Normal: N = Vec(QP) x Vec(QR)
                  = (P - Q) x (R - Q)
                  = <A, B, C>
        Plane equation: Ax + By + Cz + D = 0
        D = - (Ax_0 + By_0 + Cz_0)
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, (n x 3) - one indexed
        Output:
            - numpy ndarray of face normals
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the face normals of the mesh")
        normal_vector = np.cross(
            Remesh._get_vertices_from_idx_or_edges(
                vertices, faces[:, 1]) - Remesh._get_vertices_from_idx_or_edges(vertices, faces[:, 0]),
            Remesh._get_vertices_from_idx_or_edges(vertices, faces[:, 2]) - Remesh._get_vertices_from_idx_or_edges(vertices, faces[:, 0]))
        norm = np.linalg.norm(normal_vector, axis=-1)[:, None]
        corrected_norm = np.where(norm == 0, Remesh.__EPSILON__, norm)
        unit_normal_vector = normal_vector / corrected_norm
        return unit_normal_vector

    @staticmethod
    def _get_q_matrix(vertices: np.ndarray, faces: np.ndarray, face_normal: np.ndarray) -> np.ndarray:
        """
        Get the Q matrix of vertices.
        Q = SUM(Ki); where Ki = (p . p^T) => p represents the plane
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, (n x 3) - one indexed
            - face_normal: numpy ndarray of unit face normals
        Output:
            - numpy ndarray of Q matrix
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the Q matrix of vertices")
        # 1. Get a face mask such that a position in the mask is true if the vertex in contained in the face else false
        unique_vertices = np.unique(faces)
        # Loop edge is created to use the function _get_vertex_face_mask(), based on the way it is designed
        loop_edge = np.concatenate(
            (unique_vertices[:, None], unique_vertices[:, None]), axis=1)
        vf_mask, _ = Remesh._get_vertex_face_mask(faces, loop_edge)
        # Transpose to get (number of vertex x number of faces)
        vf_mask = vf_mask.T
        # 2. Get the 'd' of the planes
        d = -np.sum(face_normal * Remesh._get_vertices_from_idx_or_edges(vertices,
                    faces[:, 0]), axis=-1)[:, None]
        # 3. Concatenate the 'd' to the face normal
        face_normal_d = np.concatenate((face_normal, d), axis=-1)
        # 4. Use the mask to get face normals associate with each vertex
        # Broadcasting, resulting in (number of vertices x number of faces x 4)
        v = vf_mask[:, :, None] * face_normal_d[None, :, :]
        # 5. Using this, compute Q = SUM(Ki) for each vertex, calcualtion of K happens implicitly
        # (number of vertices x 4 x 4)
        Q = np.matmul(v.transpose((0, 2, 1)), v)
        return Q

    @staticmethod
    def _get_error_matrix(vertices: np.ndarray, q_matrix: np.ndarray, make_homogeneous: bool = True) -> np.ndarray:
        """
        Get the error matrix of vertices given the Q matrix.
        Delta(v) = v^T * Q * v
        Input parameters:
            - vertices: numpy ndarray of vertices, (no of vertices x 3) or (no of vertices x 4)
            - q_matrix: numpy ndarray of Q matrix, (no of vertices x 4 x 4)
        Output:
            - numpy ndarray of error matrix
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the error matrix of the vertices")
        # 1. Make the vertices homogeneous, if required
        if make_homogeneous:
            vertices_homogeneous = np.concatenate(
                (vertices, np.ones((len(vertices), 1))), axis=-1)
        else:
            vertices_homogeneous = vertices
        # 2. Compute the error matrix
        # 2.a. matmul((number of vertices x 4 x 4), (number of vertices x 4 x 1)) -> (number of vertices x 4 x 1)
        q_dot_v = np.matmul(
            q_matrix, vertices_homogeneous[:, :, None]).squeeze(axis=-1)
        # 2.b. matmul((number of vertices x 4), (number of vertices x 4)) -> (number of vertices x 1)
        E = vt_dot_q_dot_v = np.matmul(
            vertices_homogeneous[:, None, :], q_dot_v[:, :, None]).squeeze(axis=-1)
        return E

    @staticmethod
    def _get_valid_vertex_pairs(vertices: np.ndarray, incident_matrix: np.ndarray, t: int = 0) -> np.ndarray:
        """
        Get the valid vertex pairs for the given vertices and incident matrix.
        Input parameters:
            - vertices: numpy ndarray of unique vertices (n,)
            - incident_matrix: numpy ndarray of incident matrix (n x n)
            - t: integer threshold, t = 0 => edge contraction
        Output:
            - numpy ndarray of valid edges - one indexed
        """
        if t is None or t < 0:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid threshold value")
            raise ValueError("Invalid threshold value")
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the valid vertex pairs for the given vertices and incident matrix")
        incident_matrix_copy = incident_matrix.copy().astype(float)
        # 1. Get the vertex pairs that has an edge between them
        vertex_a, vertex_b = np.where(incident_matrix)
        # 2. Get the distance between the vertices: vertex_a and vertex_b
        distance = np.linalg.norm(
            vertices[vertex_a] - vertices[vertex_b], axis=-1)
        # 3. Put back the distances into the incident matrix
        incident_matrix_copy[vertex_a, vertex_b] = distance
        # 4. Use the threshold to get the valid vertex pairs
        if t > 0:
            incident_matrix_copy[incident_matrix_copy > t] = 0
        # 5. Conver the incident matrix to a boolean matrix
        incident_matrix_copy = incident_matrix_copy.astype(bool)
        # 6. Get all the valid vertex pairs
        valid_vertex_pairs = np.argwhere(incident_matrix_copy)
        # 7. Remove the duplicates
        unique_valid_vertex_pairs = np.unique(
            np.sort(valid_vertex_pairs, axis=-1), axis=0)
        # 8. Convert to edge representation - one indexed
        unique_valid_edges = unique_valid_vertex_pairs + 1
        return unique_valid_edges  # one indexed

    @staticmethod
    def _get_optimal_contractions(vertices: np.ndarray, valid_edges: np.ndarray, q_matrix: np.ndarray, heap: list = None) -> list:
        """
        Get the optimal contraction for the given valid edges and Q matrix.
        Input parameters:
            - vertices: numpy ndarray of vertices, (n x 3)
            - valid_edges: numpy ndarray of valid edges, (m x 2), one indexed
            - q_matrix: numpy ndarray of Q matrix, (n x 4 x 4)
            - heap: list of heap elements containing the error of optimal contractions
        Output:
            - list of heap elements containing the error of optimal contractions
        """
        # 1. Convert one indexed to zero indexed valid edges
        zero_indexed_valid_edges = valid_edges - 1
        # 2. For each valid edge compute the error
        if heap is None:
            heap_e = []
        else:
            heap_e = heap
        for idx, edge in enumerate(zero_indexed_valid_edges):
            # Extract the vertices and Q matrix
            v1_idx, v2_idx = edge
            v1, v2 = vertices[v1_idx], vertices[v2_idx]
            Q1, Q2 = q_matrix[v1_idx], q_matrix[v2_idx]
            # For new v_dash; Q = Q1 + Q2
            Q_v = Q1 + Q2
            # Get v_dash that minizes the error of delta(v_dash) = v_dash^T * Q * v_dash, by taking gradients
            Q = np.eye(4)
            Q[: 3] = Q_v[: 3]
            try:
                v_dash = (np.linalg.inv(Q) @
                          np.array([0, 0, 0, 1])[:, None]).T  # (1 x 4)
            except np.linalg.LinAlgError:
                logger.error(
                    f"{Remesh.__LOG_PREFIX__}: Inverse does not exist, setting to midpoint of the edge")
                v_dash = (v1 + v2) / 2
                v_dash = np.concatenate((v_dash, [1]))[None, :]  # (1 x 4)
            # Get the error
            E = Remesh._get_error_matrix(
                v_dash, Q_v[None, ...], make_homogeneous=False).squeeze().item()
            # Convert homogeneous to cartesian
            v_dash = v_dash[:, :3] / v_dash[:, 3]
            # Push the error into the heap
            heapq.heappush(heap_e, (E, (tuple(valid_edges[idx]), tuple(v_dash.squeeze()))))
        return heap_e # one indexed vertex pairs

    @staticmethod
    def _get_decimated_faces_and_edges(v1_idx: int, v2_idx: int, edges: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the decimated faces and edges for the given vertices, deleted vertices, edges and faces.
        Input parameters:
            - v1_idx: index of the first vertex, one indexed (replaced)
            - v2_idx: index of the second vertex, one indexed (deleted)
            - edges: numpy ndarray of edges, (m x 2), one indexed
            - faces: numpy ndarray of faces, (n x 3), one indexed
        Output:
            - numpy ndarray of decimated faces
            - numpy ndarray of decimated edges
        """
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Getting the decimated faces and edges for the given vertices and faces")
        # 1. Get the new edges after this decimation
        edges[edges == v2_idx] = v1_idx
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges = np.unique(np.sort(edges, axis=-1), axis=0)
        # 2. Get the new faces after this decimation
        faces[faces == v2_idx] = v1_idx
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
        _, idx = np.unique(np.sort(faces, axis=-1), axis=0, return_index=True)
        faces = faces[idx]
        return faces, edges
    
    @staticmethod
    def _set_decimated_vertices_and_faces(vertices: np.ndarray, faces: np.ndarray, deleted_vertices: set) -> Tuple[np.ndarray, np.ndarray]:
        """
        Set the decimated vertices and faces for the given vertices and faces after processing.
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, one indexed
            - deleted_vertices: set of deleted vertices
        Output: A tuple of:
            - numpy ndarray of decimated vertices
            - numpy ndarray of decimated faces
        """
        # Set vertices
        vertices_idx = np.array(list(set(np.arange(1, len(vertices)+1)) - deleted_vertices)) # get all vertices except the deleted ones; one indexed
        new_vertices = vertices[vertices_idx-1] # since, one indexed
        # Set faces
        _, inv_idx = np.unique(faces, return_inverse=True)
        new_vertices_idx = np.arange(1, len(new_vertices)+1)
        new_faces = new_vertices_idx[inv_idx].reshape(-1, 3) # since triangle mesh
        return new_vertices, new_faces

    @staticmethod
    def _decimation(vertices: np.ndarray, faces: np.ndarray, reduce_faces_to: int = 3, fixed: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Driver code for implementing decimation: apply decimation to the input mesh.
        The algorithm is implemented using matrix operations to improve performance by avoiding loops.
        Algorithm:
            1. Get the edges of the mesh
            2. Get the face normals and the constant 'd'
            3. Get the Q matrix of vertices
            4. Get the valid vertex pairs
            5. Get the optimal contractions
            6. While the number of faces is greater than the reduce_faces_to, apply decimation:
                a. Get the minimum error elements from pair contractions
                b. Update vertices and Q matrix
                c. Add the deleted vertex to the tracker set
                d. Update faces and edges
                e. Update the cost of the optimal contractions associated with `v1` the newly added vertex
        Input parameters:
            - vertices: numpy ndarray of vertices
            - faces: numpy ndarray of faces, note that the faces should be triangulated
            - reduce_faces_to: number of faces to reduce to, default is 3
            - fixed: boolean indicating whether to keep the original vertices of the mesh fixed during mesh operations
            - verbose: boolean indicating whether to make verbose
        Output:
            - vertices and faces after decimation
        """
        if vertices is None or faces is None:
            logger.error(f"{Remesh.__LOG_PREFIX__}: Invalid vertices or faces")
            raise ValueError("Invalid vertices or faces")
        if reduce_faces_to is None or reduce_faces_to < 0:
            logger.error(
                f"{Remesh.__LOG_PREFIX__}: Invalid reduce_faces_to parameter")
            raise ValueError("Invalid reduce_faces_to parameter")
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Applying decimation to the input mesh")
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Reducing the number of faces to {reduce_faces_to}")
        # Get necessary items
        og_edges, edges, face_indices, \
            unique_edges, unique_edge_indices, unique_inverse, \
            interior_edges, boundary_edges, \
            left_interior_edge_matches, right_interior_edge_matches, \
            left_boundary_edge_matches, right_boundary_edge_matches = Remesh._get_edges(
                faces)
        # Get the face normals
        face_normals = Remesh._get_face_normals_and_d(vertices, faces)
        # Get the Q matrix
        q_matrix = Remesh._get_q_matrix(vertices, faces, face_normals)
        # Get the valid vertex pairs
        incident_matrix, _, _ = get_incident_matrix(edges)
        valid_edges = Remesh._get_valid_vertex_pairs(vertices, incident_matrix)
        # Get the optimal contraction
        optimal_contractions = Remesh._get_optimal_contractions(
            vertices, valid_edges, q_matrix)
        # Initialize copies and other variables needed to track decimation
        ctr = 0
        deleted_vertices = set()
        q_matrix_copy = q_matrix.copy()
        vertices_copy, edges_copy, faces_copy = vertices.copy(), edges.copy(), faces.copy()
        new_vertices, new_faces = vertices.copy(), faces.copy()
        while len(faces_copy) > reduce_faces_to:
            logger.info(
                f"{Remesh.__LOG_PREFIX__}: Applying Decimation algorithm to the input mesh: Decimation {ctr+1} | {len(new_vertices)} Vertices | {len(new_faces)} Faces")
            # Validate triangle mesh
            if not validate_triangulated_mesh(edges_copy):
                logger.error(
                    f"{Remesh.__LOG_PREFIX__}: One or more edges are shared by more than 2 faces")
                raise ValueError(
                    "One or more edges are shared by more than 2 faces")
            # Check if the optimal contractions is empty
            if len(optimal_contractions) == 0:
                logger.info(
                    f"{Remesh.__LOG_PREFIX__}: No optimal contractions found, edges cannot be collapsed further")
                break
            # Get the minimum error elements from pair contractions
            E_min, v_info = heapq.heappop(optimal_contractions)
            v_idx, v_dash = v_info
            v1_idx, v2_idx = v_idx
            # Skip the deleted vertices
            if v1_idx in deleted_vertices or v2_idx in deleted_vertices:
                continue
            if not fixed:
                # Update vertices
                vertices_copy[v1_idx-1] = v_dash # since, v1_idx is one indexed
                # Update Q matrix
                Q1, Q2 = q_matrix_copy[v1_idx-1], q_matrix_copy[v2_idx-1] # since, v1_idx, v2_idx are one indexed
                q_matrix_copy[v1_idx-1] = Q1 + Q2
            # Add the deleted vertex to the set
            deleted_vertices.add(v2_idx) # note that, v2_idx is one indexed
            # Update faces and edges
            faces_copy, edges_copy = Remesh._get_decimated_faces_and_edges(v1_idx, v2_idx, edges_copy, faces_copy)
            # Because, we want to get the incident matrix after this decimation without losing the original vertex information
            # and to maintain the original vertex indices, we do the following
            proxy_edges = np.array([[i, i] for i in deleted_vertices])
            proxy_edge_copy = np.concatenate((edges_copy, proxy_edges), axis=0)
            incident_matrix, _, _ = get_incident_matrix(proxy_edge_copy, clean=True)
            # Update the cost of the optimal contractions associated with v1_idx
            associated_vertices = np.where(incident_matrix[v1_idx-1])[0] + 1 # since, we want the indexing to be one indexed
            v1_valid_pairs = np.vstack(
                ([v1_idx]*len(associated_vertices), associated_vertices)).T
            optimal_contractions = Remesh._get_optimal_contractions(
                vertices_copy, v1_valid_pairs, q_matrix_copy, heap=optimal_contractions)
            # Plot for visualization
            if verbose and ctr % Remesh.__PLOT_EVERY__ == 0:
                Remesh._plot(vertices, unique_edges, vertices_copy, p_alpha=0.15,
                             title=f"Decimation: Decimation {ctr+1} - Decimated vertices")
                Remesh._plot_faces(
                    vertices, faces, title=f"3D Plot - Original Faces at the start of this decimation | {len(faces)} Faces")
                Remesh._plot_faces(
                    vertices_copy, faces_copy, title=f"3D Plot - decimation: Decimation {ctr+1} - New Faces | {len(faces_copy)} Faces", save=True, file_name=f"decimation_{ctr+1}")
            # Update the face if there is only one face for visualization
            # Trimesh workflow
            if len(faces_copy) == 1:
                faces_copy = np.concatenate((faces_copy, faces_copy[:, ::-1]))
            # Get the cleaned vertices and faces after decimation
            new_vertices, new_faces = Remesh._set_decimated_vertices_and_faces(vertices_copy, faces_copy, deleted_vertices)
            # Increment the counter
            ctr += 1
        logger.info(
            f"{Remesh.__LOG_PREFIX__}: Decimation algorithm resulted in: {len(new_vertices)} Vertices | {len(new_faces)} Faces")
        return new_vertices, new_faces
