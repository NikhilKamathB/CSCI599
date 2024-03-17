##########################################################################################################
# Assignment 1: Mesh Subdivision and Decimation
# How to run this code:
#     - Open a terminal and navigate to the directory containing this file.
#     - Run the following command: `python assignment1.py`
##########################################################################################################

# Set system path to include the parent directory
import os
import argparse
import sys
sys.path.append('..')

# Import the required modules
import src
import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest


def subdivision_loop(mesh, iterations: int = 1, fixed: bool = False, verbose: bool = True, output_dir: str = '../assets/results/assignment1'):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :param fixed: boolean indicating whether to keep the original vertices of the mesh fixed during mesh operations
    :param verbose: boolean indicating whether to make verbose
    :param output_dir: path to the output directory in which .obj file will be stored
    :return: mesh after subdivision
    
    Overall process:
    Reference: https://github.com/mikedh/trimesh/blob/main/trimesh/remesh.py#L207
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.
    
    # The following implementation considers only the interior cases
    # You should also consider the boundary cases and more iterations in your submission
    """
    new_vertices, new_faces = mesh.loop_subdivision(iterations=iterations, fixed=fixed, verbose=verbose)
    mesh.save(new_vertices, new_faces, os.path.join(output_dir, 'cube_subdivided.obj'))
    return mesh


def simplify_quadric_error(mesh, face_count: int = 1, fixed: bool = False, verbose: bool = True, output_dir: str = '../assets/results/assignment1'):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :param fixed: boolean indicating whether to keep the original vertices of the mesh fixed during mesh operations
    :param verbose: boolean indicating whether to make verbose
    :param output_dir: path to the output directory in which .obj file will be stored
    :return: mesh after decimation
    """
    new_vertices, new_faces = mesh.decimation(faces=face_count, fixed=fixed, verbose=verbose)
    mesh.save(new_vertices, new_faces, os.path.join(output_dir, 'cube_decimated.obj'))
    return mesh


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Mesh operations - Loop subdivision and Decimation.")
    parser.add_argument("-i", "--obj_file", type=str, default='../assets/cube.obj', help="Path to the .obj file")
    parser.add_argument("-o", "--output_dir", type=str, default='../assets/results/assignment1', help="Path to the output directory in which .obj file will be stored")
    parser.add_argument("-s", "--subdivision", type=int, default=4, help="Number of iterations for Loop subdivision")
    parser.add_argument("-d", "--decimation", type=int, default=5, help="Number of faces desired in the resulting mesh after decimation")
    parser.add_argument("-f", "--fixed", type=src.str2bool, default='n', help="Keep the original vertices of the mesh fixed during mesh operations")
    parser.add_argument("-v", "--verbose", type=src.str2bool, default='y', help="Make verbose")
    args = parser.parse_args()
    
    # Global variables
    OBJ_FILE = args.obj_file
    OUTPUT_DIR = args.output_dir
    SUBDIVISION = args.subdivision
    DECIMATION = args.decimation
    FIXED = args.fixed
    VERBOSE = args.verbose

    # Make the output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ############################################################
    ########### Assignment 1. Trimesh implementation ###########
    ############################################################

    # try:
    #   mesh = trimesh.creation.box(extents=[2, 2, 2])

    #   mesh_subdivided = mesh.subdivide_loop(iterations=SUBDIVISION)
    #   mesh_subdivided.export(os.path.join(OUTPUT_DIR, 'cube_subdivided_trimesh.obj'))

    #   mesh_decimated = mesh.simplify_quadric_decimation(DECIMATION)
    #   mesh_decimated.export(os.path.join(OUTPUT_DIR, 'cube_decimated_trimesh.obj'))
    # except Exception as e:
    #   raise e

    ############################################################
    
    ############################################################
    ########### Assignment 1. Custom implementation ############
    ############################################################

    try:
        mesh = src.load_3d_file(OBJ_FILE)
        _ = subdivision_loop(mesh, iterations=SUBDIVISION, fixed=FIXED, verbose=VERBOSE, output_dir=OUTPUT_DIR)
        _ = simplify_quadric_error(mesh, face_count=DECIMATION, fixed=FIXED, verbose=VERBOSE, output_dir=OUTPUT_DIR)
    except Exception as e:
        raise e
    
    ############################################################

