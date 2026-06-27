# REQUIRED PACKAGES
# pip install numpy-stl

import numpy as np
import pandas as pd
import scipy
from src.robotics import quat2pose, interpT, inverse_T

def load_pose_data_from_csv(csv_file):
    """
    This function loads the .csv file in the data folder.
    It is assumed that this .csv file has the following format:
    - it has no header row
    - its first column contains the time vector
    - its second to fourth columns contain the xyz position coordinates
    - its fifth to eighth columns contain the quaternion coordinates [w x y z]
    It returns the pose trajectory T, and the time vector t
    """

    raw_data = pd.read_csv(csv_file, header=None, dtype=np.float64).values

    t = raw_data[:, 0]
    pos = raw_data[:, 1:4].T
    quat = raw_data[:, 4:8].T

    T = quat2pose(pos, quat)

    return T, t

def remove_offset_array(array):
    array -= array[0]
    return array

def calculate_number_of_equidistant_steps_in_array(array, stepsize = 0.02):
    array_without_offset = remove_offset_array(array)
    N = int(1 + np.floor(array_without_offset[-1]/stepsize))
    return N
    
def make_array_equidistant(array, N):
    array_equidistant = np.linspace(array[0],array[-1],N)
    return array_equidistant

def write_ndarray_to_csv_file(ndarray, file_location):
    
    df = pd.DataFrame(ndarray)
    df.to_csv(file_location, index = False, header = False, float_format='%.5g', compression='gzip')
    
    return

def load_demo_trajectory(input_trajectory,path_to_data):
    if input_trajectory == 'helical_translation':
        T, dt = synthetic_helical_translation()
    elif  input_trajectory == 'axis_rotation':
        T, dt = synthetic_axis_rotation()
    elif  input_trajectory == 'precession':
        T, dt = synthetic_precession()
    elif input_trajectory == 'pouring':
        T, dt = load_recorded_pouring_motion(path_to_data)
    N = T.shape[2]
    time_total = (N-1)*dt
    return T, N, dt, time_total


def synthetic_helical_translation():
    """
    Generate trajectory data of a helical translation
    """
    N = 30 # number of samples
    time_total = 2 # seconds
    time_axis = np.linspace(0, time_total, N)
    dt = time_total/(N-1)  # [s]

    r = 0.2  # radius of the circular trajectory
    p_x = np.array([r * np.cos(np.pi * time_axis/time_total)])
    p_y = np.array([r * np.sin(np.pi * time_axis/time_total)])
    p_z = np.array([r*time_axis/time_total])
    p = np.vstack([p_x, p_y, p_z])
    p += 1e-5 * np.random.randn(3, N)

    T = np.zeros((4, 4, N))
    for k in range(N):
        T[0:3, 3, k] = p[:, k]
        T[0:3, 0:3, k] = scipy.spatial.transform.Rotation.from_euler('xyz', 1e-5 * np.random.randn(3)).as_matrix()
        T[3, 3, k] = 1
    
    return T, dt

def synthetic_axis_rotation():
    """
    Generate trajectory data of a rotation about a fixed axis
    """
    N = 60
    time_total = 4 # seconds
    time_axis = np.linspace(0, time_total, N)
    dt = time_total/(N-1)  # [s]
    
    T = np.zeros((4, 4, N))
    for k in range(N):
    
        ROT = scipy.spatial.transform.Rotation.from_euler('z', 270 * time_axis[k]/time_total, degrees=True).as_matrix()
        T[0:3, 0:3, k] = ROT
        T[3, 3, k] = 1

        # displace body frame origin away from the zero vector
        T_disp = np.eye(4)
        T_disp[0,3] = 0.2
        T[: ,: , k] = T[: ,: , k] @ T_disp

        # Add artifical noise to avoid exact singularities
        T[0:3, 3, k] = T[0:3, 3, k] + 1e-5 * np.random.randn(1,3)
        T[0:3, 0:3, k] = T[0:3, 0:3, k] @ scipy.spatial.transform.Rotation.from_euler('xyz', 1e-5 * np.random.randn(3)).as_matrix()
    
    return T, dt

def synthetic_precession():
    """
    Generate trajectory data of a precession motion
    """
    N = 60
    time_total = 4 # seconds
    time_axis = np.linspace(0, time_total, N)
    dt = time_total/(N-1)  # [s]
    
    T = np.zeros((4, 4, N))
    for k in range(N):
        
        # First rotation
        ROT1 = scipy.spatial.transform.Rotation.from_euler('z', 270 * time_axis[k]/time_total, degrees=True).as_matrix()
        T[0:3, 0:3, k] = ROT1
        T[3, 3, k] = 1

        # Displace body frame origin away from the zero vector
        T_disp = np.eye(4)
        T_disp[0,3] = 0.2
        T[: ,: , k] = T[: ,: , k] @ T_disp

        # Second rotation -> precession
        ROT2 = scipy.spatial.transform.Rotation.from_euler('x', 270 * time_axis[k]/time_total, degrees=True).as_matrix()
        T[0:3, 0:3, k] = T[0:3, 0:3, k] @ ROT2
        
        # Add artifical noise to avoid exact singularities
        T[0:3, 3, k] = T[0:3, 3, k] + 1e-5 * np.random.randn(1,3)
        T[0:3, 0:3, k] = T[0:3, 0:3, k] @ scipy.spatial.transform.Rotation.from_euler('xyz', 1e-5 * np.random.randn(3)).as_matrix()
    
    return T, dt

def load_recorded_pouring_motion(path_to_data):

    data_file = rf'{path_to_data}/Demos/pouring/Trial1_coffee_kettle_ref_top.csv'
    T, t = load_pose_data_from_csv(data_file)
    t = remove_offset_array(t)

    # Interpolate pose data to equidistant timesteps
    dt = 0.02
    N = calculate_number_of_equidistant_steps_in_array(t, stepsize = dt)
    t_equidistant = make_array_equidistant(t, N)
    T = interpT(t, T, t_equidistant)

    # Extract first half of the trajectory
    index_cut = round(N / 2) + 20
    T = T[:, :, :index_cut]  

    return T, dt

def compute_normals(polygons):
    normals = []
    for polygon in polygons:
        v1, v2, v3 = polygon[0,:], polygon[1,:], polygon[2,:]
        normal = np.cross(v2 - v1, v3 - v1)
        normal /= np.linalg.norm(normal) + 1e-10
        normals.append(normal)
    return np.array(normals)

def construct_polygons(vertices, faces):
    polygons = np.array([[vertices[j,:] for j in face] for face in faces])
    return polygons

def load_obj_file(filepath):
    
    vertices = []
    faces = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v'):
                vertex = [float(x)/1000.0 for x in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('f'):
                face = [int(x)-1 for x in line.strip().split()[1:]]
                faces.append(face)

    return {
        'faces': np.array(faces),
        'vertices': np.array(vertices),
    }

def create_cube_data():
    
    # Define cube vertices
    r = [-0.07, 0.07]  
    vertices = np.array([
        [0.05 + r[0], r[0]-0.01, r[0]], [0.05 + r[1], r[0]-0.01, r[0]],
        [0.05 + r[1], r[1]-0.01, r[0]], [0.05 + r[0], r[1]-0.01, r[0]],
        [0.05 + r[0], r[0]-0.01, r[1]], [0.05 + r[1], r[0]-0.01, r[1]],
        [0.05 + r[1], r[1]-0.01, r[1]], [0.05 + r[0], r[1]-0.01, r[1]]
    ])

    # Define cube faces (each face is a list of 4 vertices)
    faces = np.array([
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [4, 7, 3, 0]   # left
    ])
    
    return {
        'faces': faces,
        'vertices': vertices,
    }

def scale_object_vertices(object, scale = 1.0):
    object['vertices'] = object['vertices']*scale
    return object

def translate_object_vertices(object,translation):
    object['vertices'] += np.array(translation)
    return object

def rotate_object_vertices(object,rotation):
    object['vertices'] = object['vertices'] @ rotation.T
    return object

def load_data_kettle(path_to_data):
    kettle = load_obj_file(rf"{path_to_data}/Demos/pouring_objects/kettle.obj")
    return kettle

def load_tracker_kettle_calibration_data():
    # Retrieve the pose of the base frame of the kettle CAD model w.r.t. the tracker frame
    T_tracker_wrt_kettle = np.eye(4)
    T_tracker_wrt_kettle[:3,3] = np.array([0., 0.1, 0.04])
    T_tracker_wrt_kettle[:3,:3] = scipy.spatial.transform.Rotation.from_euler('YZY', [180, -90, 0], degrees=True).as_matrix()
    T_kettle_wrt_tracker = inverse_T(T_tracker_wrt_kettle)
    return T_kettle_wrt_tracker

def read_descriptor(location):
    descriptor = pd.read_csv(location,header=None, compression='gzip').values
    return descriptor



