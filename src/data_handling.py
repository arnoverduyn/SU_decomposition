# REQUIRED PACKAGES
# pip install numpy-stl

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from src.robotics import quat2pose, interpT

def load_csv_file(csv_file):
    """
    This function loads the .csv file in the data folder.
    It is assumed that this .csv file has the following format:
    - its first column contains the time vector
    - its second to fourth columns contain the xyz position coordinates
    - its fifth to eighth columns contain the quaternion coordinates [w x y z]
    """
    dtypes = {'col1': 'float64', 'col2': 'float64', 'col3': 'float64', 'col4': 'float64', 'col5': 'float64', 'col6': 'float64', 'col7': 'float64', 'col8': 'float64'}  
    raw_data = pd.read_csv(csv_file, header=None, dtype=dtypes).values
    time_vector = raw_data[:, 0]
    time_vector -= time_vector[0]

    pos = raw_data[:, 1:4].T
    quat = raw_data[:, 4:8].T
    T = quat2pose(pos, quat)

    # Resample to an equidistant sampling interval (50Hz)
    dt = 0.02  # [s]
    N = int(1 + np.floor(time_vector[-1]/dt))
    time_new = np.linspace(0,time_vector[-1],N)
    
    T = interpT(time_vector, T, time_new)
    
    return T, dt

def write_csv_file(file, file_location):
    
    df = pd.DataFrame(file)
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
    return T, dt


def synthetic_helical_translation():
    """
    Generate trajectory data of a helical translation
    """
    N = 60 # number of samples
    time_total = 1 # seconds
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
        T[0:3, 0:3, k] = R.from_euler('xyz', 1e-5 * np.random.randn(3)).as_matrix()
        T[3, 3, k] = 1
    
    return T, dt

def synthetic_axis_rotation():
    """
    Generate trajectory data of a rotation about a fixed axis
    """
    N = 60
    time_total = 2 # seconds
    time_axis = np.linspace(0, time_total, N)
    dt = time_total/(N-1)  # [s]
    
    T = np.zeros((4, 4, N))
    for k in range(N):
    
        ROT = R.from_euler('z', 270 * time_axis[k]/time_total, degrees=True).as_matrix()
        T[0:3, 0:3, k] = ROT
        T[3, 3, k] = 1

        # displace body frame origin away from the zero vector
        T_disp = np.eye(4)
        T_disp[0,3] = 0.2
        T[: ,: , k] = T[: ,: , k] @ T_disp

        # Add artifical noise to avoid exact singularities
        T[0:3, 3, k] = T[0:3, 3, k] + 1e-5 * np.random.randn(1,3)
        T[0:3, 0:3, k] = T[0:3, 0:3, k] @ R.from_euler('xyz', 1e-5 * np.random.randn(3)).as_matrix()
    
    return T, dt

def synthetic_precession():
    """
    Generate trajectory data of a precession motion
    """
    N = 60
    time_total = 2 # seconds
    time_axis = np.linspace(0, time_total, N)
    dt = time_total/(N-1)  # [s]
    
    T = np.zeros((4, 4, N))
    for k in range(N):
        
        # First rotation
        ROT1 = R.from_euler('z', 270 * time_axis[k]/time_total, degrees=True).as_matrix()
        T[0:3, 0:3, k] = ROT1
        T[3, 3, k] = 1

        # Displace body frame origin away from the zero vector
        T_disp = np.eye(4)
        T_disp[0,3] = 0.2
        T[: ,: , k] = T[: ,: , k] @ T_disp

        # Second rotation -> precession
        ROT2 = R.from_euler('x', 270 * time_axis[k]/time_total, degrees=True).as_matrix()
        T[0:3, 0:3, k] = T[0:3, 0:3, k] @ ROT2
        
        # Add artifical noise to avoid exact singularities
        T[0:3, 3, k] = T[0:3, 3, k] + 1e-5 * np.random.randn(1,3)
        T[0:3, 0:3, k] = T[0:3, 0:3, k] @ R.from_euler('xyz', 1e-5 * np.random.randn(3)).as_matrix()
    
    return T, dt

def load_recorded_pouring_motion(path_to_data):
    data_file = rf'{path_to_data}/Demos/pouring/Trial1_coffee_kettle_ref_top.csv'
    T, dt = load_csv_file(data_file)
    N = T.shape[2]
    T = T[:, :, :round(N / 2) + 20]  # Cut to first half
    return T, dt/2.0

def retrieve_data_designed_objects(path_to_data):
    kettle = load_object(rf"{path_to_data}/Demos/pouring_objects/kettle.obj", scale=1/1500, translation=[0, 0, -0.02], rotation=R.from_euler('z', -125, degrees=True))
    bottle = load_object(rf"{path_to_data}/Demos/pouring_objects/bottle.obj", scale=1/1500, translation=[0, 0, -0.18], rotation=R.from_euler('z', -90, degrees=True))
    table = load_object(rf"{path_to_data}/Demos/pouring_objects/table.obj", scale=1/1000, translation=[-0.3, -0.05, -0.02])
    cup = load_object(rf"{path_to_data}/Demos/pouring_objects/cup.obj", scale=1/2000, translation=[-0.54, -0.09, -0.02])

    kettle['color'] = [0, 0.5, 1]

    return kettle, bottle, table, cup

def load_object(filepath, scale=1.0, translation=None, rotation=None):
    
    vertices = []
    faces = []
    
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = [float(x) for x in line.strip().split()[1:]]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(x.split('/')[0]) - 1 for x in line.strip().split()[1:]]
                faces.append(face)
    vertices = np.array(vertices) * scale 
    faces = np.array(faces) 
    
    if translation is not None:
        vertices += np.array(translation)
    if rotation is not None:
        vertices = rotation.apply(vertices)
    vertices = np.vstack([vertices.T, np.ones(vertices.shape[0])])

    return {
        'faces': faces,
        'number_of_vertices': vertices.shape[1],
        'homogeneous_vertices': vertices,
    }

def read_descriptor(location):
    descriptor = pd.read_csv(location,header=None, compression='gzip').values
    return descriptor



