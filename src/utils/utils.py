###################################################################################################
# Miscelaneous utilities for the project goes here
###################################################################################################

import os
import numpy as np
from typing import Tuple


def intify(item: list) -> list:
    """
    Convert a list of strings to a list of integers.
    Input parameters:
        - item: list of strings
    Output:
        - list of integers
    """
    return list(map(int, item))

def floatify(item: list) -> list:
    """
    Convert a list of strings to a list of floats.
    Input parameters:
        - item: list of strings
    Output:
        - list of floats
    """
    return list(map(float, item))

def convert_to_numpy_matrix(data: list) -> np.ndarray:
    """
        Converts list to a numpy array.
        Input parameters:
            - data: list of items
        Output:
            - numpy array of dtype float32
    """
    return np.asarray(data, dtype=np.float32)

def get_edges_from_faces_vstack(faces: np.ndarray) -> np.ndarray:
    """
    Get the edges of the mesh given a list of faces.
    Input parameters:
        - faces: numpy ndarray of faces (n x 3)
    Output:
        - numpy ndarray of edges (m x 2)
    """
    return np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))

def get_edges_from_faces_hstack(faces: np.ndarray) -> np.ndarray:
    """
    Get the edges of the mesh given a list of faces.
    Input parameters:
        - faces: numpy ndarray of faces (n x 3)
    Output:
        - numpy ndarray of edges (m x 2)
    """
    return np.hstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))

def get_special_edges(unique_edges: np.ndarray, unique_inverse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the interior and boundary edges of the mesh given a set of unique edges and unique inverse.
    Input parameters:
        - unique_edges: numpy ndarray of unique edges (p x 2)
        - unique_inverse: numpy ndarray of unique inverse (m,)
    Output:
        - interior_edges: numpy ndarray of interior edges (x, 2)
        - boundary_edges: numpy ndarray of boundary edges (y, 2)
    """
    interior_edges = unique_edges[np.bincount(unique_inverse) == 2]
    boundary_edges = unique_edges[np.bincount(unique_inverse) == 1]
    return interior_edges, boundary_edges

def get_matching_row_indices(a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the matching row indices of two numpy arrays.
    Input parameters:
        - a1: numpy ndarray (m x 3)
        - a2: numpy ndarray (n x 3)
    Output:
        - a tuple of numpy arrays representing the matching row indices in a1 (q,) and a2 (q,)
    """
    a1_match, a2_match = np.where(np.all(a1[:, np.newaxis] == a2, axis=-1))
    return a1_match, a2_match

def get_edges(faces: np.ndarray) -> Tuple[np.ndarray, ...]:
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
    edges, og_edges = get_edges_from_faces_vstack(faces), get_edges_from_faces_hstack(faces)
    edges.sort(axis=1)
    unique_edges, unique_edge_indices, unique_inverse = np.unique(edges, axis=0, return_index=True, return_inverse=True)
    face_indices = np.tile(np.arange(len(faces)), 3)
    interior_edges, boundary_edges = get_special_edges(unique_edges, unique_inverse)
    left_interior_edge_matches, right_interior_edge_matches = get_matching_row_indices(edges, interior_edges)
    left_boundary_edge_matches, right_boundary_edge_matches = get_matching_row_indices(edges, boundary_edges)
    return og_edges, edges, face_indices, \
        unique_edges, unique_edge_indices, unique_inverse, \
        interior_edges, boundary_edges, \
        left_interior_edge_matches, right_interior_edge_matches, \
        left_boundary_edge_matches, right_boundary_edge_matches

def validate_triangulated_mesh(edges: np.ndarray) -> bool:
    """
    Validate that the mesh obeys traingulation rules:
        - no edge is shared by more than 2 faces
    Input parameters:
        - edges: numpy ndarray of edges (m x 2)
    Output:
        - boolean indicating whether the mesh is triangulated
    """
    _, count_array = np.unique(edges, axis=0, return_counts=True)
    return np.all(count_array <= 2)


def get_incident_matrix(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the incident matrix of the mesh given a list of edges.
    Input parameters:
        - edges: numpy ndarray of unique edges (m x 2)
    Output:
        - a tuple of numpy arrays representing the incident matrix (n x n), unique vertices (n,) and total neighbours (n,)
    """
    vertices, inverse = np.unique(edges, return_inverse=True)
    inverse = inverse.reshape((-1, 2)) # Reshape to (m, 2)
    incident_matrix = np.zeros((len(vertices), len(vertices)), dtype=bool)
    incident_matrix[inverse[:, 0], inverse[:, 1]] = True
    incident_matrix[inverse[:, 1], inverse[:, 0]] = True
    neighbours = incident_matrix.sum(axis=-1)
    return incident_matrix, vertices, neighbours

def save_obj(vertices: np.ndarray, faces: np.ndarray, file_path: str) -> None:
    """
    Save the vertices and faces to a .obj file.
    Input parameters:
        - vertices: numpy array of vertices
        - faces: numpy array of faces
        - file_path: path to the file
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {' '.join(map(str, vertex))}\n")
        for face in faces:
            file.write(f"f {' '.join(map(str, face))}\n")
