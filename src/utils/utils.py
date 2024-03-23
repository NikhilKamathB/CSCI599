###################################################################################################
# Miscelaneous utilities for the project goes here
###################################################################################################

import os
import cv2
import numpy as np
from typing import Tuple


def str2bool(v) -> bool:
    '''
        Convert string to boolean, basically used by the cmd parser.
    '''
    return v.lower() in ("yes", "Yes", "YES", "y", "true", "True", "TRUE", "t", "1")

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

def get_incident_matrix(edges: np.ndarray, clean: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the incident matrix of the mesh given a list of edges.
    Input parameters:
        - edges: numpy ndarray of unique edges (m x 2)
        - clean: boolean indicating whether to clean the incident matrix after creation, all clean operations can be done within the block
    Output:
        - a tuple of numpy arrays representing the incident matrix (n x n), unique vertices (n,) and total neighbours (n,)
    """
    vertices, inverse = np.unique(edges, return_inverse=True)
    inverse = inverse.reshape((-1, 2)) # Reshape to (m, 2)
    incident_matrix = np.zeros((len(vertices), len(vertices)), dtype=bool)
    incident_matrix[inverse[:, 0], inverse[:, 1]] = True
    incident_matrix[inverse[:, 1], inverse[:, 0]] = True
    if clean:
        np.fill_diagonal(incident_matrix, False)
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

def serialize_keypoints(keypoints: tuple) -> list:
    """
        Serialize the keypoints.
        Input parameters:
            - keypoints: keypoints to serialize
        Output:
            - serialized keypoints
    """
    out = []
    for kp in keypoints:
        out.append([kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id])
    return out

def serialize_matches(matches: list) -> list:
    """
        Serialize the matches.
        Input parameters:
            - matches: matches to serialize
        Output:
            - serialized matches
    """
    out = []
    for match in matches:
        out.append([match.queryIdx, match.trainIdx, match.imgIdx, match.distance])
    return out

def deserialize_keypoints(keypoints: list) -> list:
    """
        Deserialize the keypoints.
        Input parameters:
            - keypoints: keypoints to deserialize
        Output:
            - deserialized keypoints
    """
    out = []
    for keypoint in keypoints:
        out.append(
            cv2.KeyPoint(x=keypoint[0][0], y=keypoint[0][1], size=keypoint[1], angle=keypoint[2], response=keypoint[3], octave=keypoint[4], class_id=keypoint[5])
        )
    return out

def deserialize_matches(matches: list) -> list:
    """
        Deserialize the matches.
        Input parameters:
            - matches: matches to deserialize
        Output:
            - deserialized matches
    """
    out = []
    for match in matches:
        out.append(
            cv2.DMatch(queryIdx=match[0], trainIdx=match[1], imgIdx=match[2], distance=match[3])
        )
    return out
