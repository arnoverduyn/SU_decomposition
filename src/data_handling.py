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
    if input_trajectory == 'translation':
        T, dt = synthetic_pose_trajectory()
    elif  input_trajectory == 'rotation':
        T, dt = synthetic_pose_trajectory_rot()
    elif input_trajectory == 'pouring':
        T, dt = load_recorded_pouring_motion_gen(path_to_data)
    return T, dt


def synthetic_pose_trajectory():
    """
    Generate synthetic trajectory
    """
    N = 30
    s = np.linspace(0, 1, N)
    r = 0.2  # 20 cm
    p_x = np.concatenate([r * np.cos(np.pi * s), -3/2 * r + r/2 * np.cos(np.pi * s[1:])])
    p_y = np.concatenate([r * np.sin(np.pi * s), r/2 * np.sin(np.pi * (1 + s[1:]))])
    p_z = np.concatenate([np.zeros_like(s), np.zeros_like(s[1:])])
    p = np.vstack([1 + p_x, 1 + p_y, p_z])
    N = p.shape[1]
    p += 1e-4 * np.random.randn(3, N)
    
    # Perform a smoothing action
    p = np.transpose(pd.DataFrame(p).T.rolling(window=2, min_periods=1).mean().values)

    T = np.zeros((4, 4, N))
    for k in range(N):
        T[0:3, 3, k] = p[:, k]
        T[0:3, 0:3, k] = R.from_euler('xyz', 1e-3 * np.random.randn(3)).as_matrix()
        T[3, 3, k] = 1
    dt = 0.05  # [s]
    
    return T, dt

def synthetic_pose_trajectory_rot():
    """
    Generate synthetic trajectory
    """
    N = 60
    s = np.linspace(0, 1, N)
    r = 0.2  # 20 cm
    p_x = r * np.cos(np.pi * s)
    p_y = r * np.sin(np.pi * s)
    p_z = np.zeros_like(s)
    p = np.vstack([1 + p_x, 1 + p_y, p_z])
    N = p.shape[1]
    p += 1e-4 * np.random.randn(3, N)
    
    # Perform a smoothing action
    p = np.transpose(pd.DataFrame(p).T.rolling(window=2, min_periods=1).mean().values)

    T = np.zeros((4, 4, N))
    for k in range(N):
        T[0:3, 3, k] = p[:, k]
        T[0:3, 0:3, k] = (R.from_euler('xyz', 1e-3 * np.random.randn(3)) *
                          R.from_euler('z', 180 * s[k], degrees=True)).as_matrix()
        T[3, 3, k] = 1
    dt = 0.05  # [s]
    
    return T, dt

def load_recorded_pouring_motion_gen(path_to_data):
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



