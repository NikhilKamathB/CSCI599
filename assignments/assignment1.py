##########################################################################################################
# Assignment 1: Mesh Subdivision and Decimation
# How to run this code:
#     - Open a terminal and navigate to the directory containing this file.
#     - Run the following command: `python assignment1.py`
##########################################################################################################

# Set system path to include the parent directory
import sys
sys.path.append('..')

# Import the required modules
import src
import trimesh


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    new_vertices, new_faces = mesh.loop_subdivision(iterations=iterations)
    return new_vertices, new_faces


def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    return mesh


if __name__ == '__main__':
    # Load mesh and print information
    # mesh = trimesh.load_mesh('../assets/cube.obj')
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    # print(f'Mesh Info: {mesh}')

    # apply loop subdivision over the loaded mesh
    mesh_subdivided = mesh.subdivide_loop(iterations=3)
    # mesh_subdivided.show()
    # import pdb; pdb.set_trace()

    # TODO: implement your own loop subdivision here
    # mesh_subdivided = subdivision_loop(mesh, iterations=2)

    # print the new mesh information and save the mesh
    # print(f'Subdivided Mesh Info: {mesh_subdivided}')
    # mesh_subdivided.export('../assets/assignment1/cube_subdivided.obj')
    # mesh_subdivided.export('temp2.obj')
    # import sys
    # sys.exit(0)

    # quadratic error mesh decimation
    # mesh_decimated = mesh.simplify_quadric_decimation(4)

    # TODO: implement your own quadratic error mesh decimation here
    # mesh_decimated = simplify_quadric_error(mesh, face_count=1)

    # print the new mesh information and save the mesh
    # print(f'Decimated Mesh Info: {mesh_decimated}')
    # mesh_decimated.export('../assets/assignment1/cube_decimated.obj')

    # Global variables
    OBJ_FILE = '../assets/cube.obj'
    # Load mesh
    try:
        mesh = src.load_3d_file(OBJ_FILE)
        # TODO: implement your own loop subdivision here
        mesh_subdivided_vertices, mesh_subdivided_faces = subdivision_loop(mesh, iterations=3)
        mesh.save(mesh_subdivided_vertices, mesh_subdivided_faces, '../assets/results/assignment1/cube_subdivided.obj')
    except Exception as e:
        # # To get the stack trace, use the following line instead:
        raise e
        # print(f"An Error occurred: {e}")
