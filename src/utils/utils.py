###################################################################################################
# Miscelaneous utilities for the project goes here
###################################################################################################

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

def get_special_edges(unique_edges: np.ndarray, unique_inverse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the interior and boundary edges of the mesh given a set of unique edges and unique inverse.
    Input parameters:
        - unique_edges: numpy ndarray of unique edges (p x 2)
        - unique_inverse: numpy ndarray of unique inverse (m,)
    Output:
        - interior_edges: numpy ndarray of interior edges
        - boundary_edges: numpy ndarray of boundary edges
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
        - edges: numpy ndarray of edges (m x 2)
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
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    unique_edges, unique_edge_indices, unique_inverse = np.unique(edges, axis=0, return_index=True, return_inverse=True)
    face_indices = np.tile(np.arange(len(faces)), 3)
    interior_edges, boundary_edges = get_special_edges(unique_edges, unique_inverse)
    left_interior_edge_matches, right_interior_edge_matches = get_matching_row_indices(edges, interior_edges)
    left_boundary_edge_matches, right_boundary_edge_matches = get_matching_row_indices(edges, boundary_edges)
    return edges, face_indices, \
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