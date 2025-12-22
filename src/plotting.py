import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation as R
import src.data_handling
    
# Normals for shading
def compute_normals(faces):
    normals = []
    for face in faces:
        v1, v2, v3 = face[0], face[1], face[2]
        normal = np.cross(v2 - v1, v3 - v1)
        normal /= np.linalg.norm(normal) + 1e-8
        normals.append(normal)
    return np.array(normals)


def plot_trajectories(T_sub, T_var, input_trajectory, path_to_data, path_to_figures):
    if input_trajectory == 'pouring':
        (kettle,_,_,_) = src.data_handling.retrieve_data_designed_objects(path_to_data)
        plot_trajectories_with_kettle(T_sub, T_var, kettle, path_to_figures)
    else:
        plot_trajectories_with_cube(T_sub, T_var, path_to_figures) 


def plot_trajectories_with_cube(T, T_var, path_to_figures):
    
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Lighting setup
    ls = LightSource(azdeg=135, altdeg=45)

    N = T.shape[2]
    keyframes = [0, round(0.22* N ), round(0.42* N ), round(0.6* N ), N-1]

    
    # Define cube vertices
    r = [-0.07, 0.07]  # cube from -0.1 to 0.1 in all axes
    vertices = np.array([
        [0.05 + r[0], r[0]-0.01, r[0], 1.], [0.05 + r[1], r[0]-0.01, r[0], 1.],
        [0.05 + r[1], r[1]-0.01, r[0], 1.], [0.05 + r[0], r[1]-0.01, r[0], 1.],
        [0.05 + r[0], r[0]-0.01, r[1], 1.], [0.05 + r[1], r[0]-0.01, r[1], 1.],
        [0.05 + r[1], r[1]-0.01, r[1], 1.], [0.05 + r[0], r[1]-0.01, r[1], 1.]
    ])
    vertices = vertices.T

    # Define cube faces (each face is a list of 4 vertices)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [4, 7, 3, 0]   # left
    ]

    for k in keyframes:

        transformed_vertices = T[:, :, k] @ vertices

        # Face vertices for 3D collection
        poly3d = [[transformed_vertices[:3, j] for j in face] for face in faces]
        verts_array = np.array([np.array(face) for face in poly3d])

        normals = compute_normals(verts_array)

        # Lighting shading via LightSource
        shade_vals = ls.shade_normals(normals, fraction=1.0)  # Grayscale
        cmap = plt.cm.Blues  # Or try 'viridis', 'plasma', etc.
        shade_colors = cmap(0.4/(shade_vals+0.01))  # Map grayscale to RGBA
        collection = Poly3DCollection(verts_array, facecolors=shade_colors, edgecolors='none', linewidths=0)
        collection.set_alpha(0.15)
        ax.add_collection3d(collection)

    # Plot trajectories of the origins of the different frames
    nb_lines = len(T_var)
    colors = ['b', 'r']

    for j in range(nb_lines):
        p = T_var[j][0:3, 3, :] 
        ax.plot(p[0, :], p[1, :], p[2, :], color=colors[j], linewidth=4)

        # Plot orientation body frame
        start_end_frames = [keyframes[0],keyframes[-1]]
        ROT = T_var[j][:3,:3,start_end_frames]

        arrow_len = 0.08  
        ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 0], ROT[1, 0], ROT[2, 0], length=arrow_len, normalize=True, color=colors[j], linewidth=3.0)
        ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 1], ROT[1, 1], ROT[2, 1], length=arrow_len, normalize=True, color=colors[j], linewidth=3.0)
        ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 2], ROT[1, 2], ROT[2, 2], length=arrow_len, normalize=True, color=colors[j], linewidth=3.0)

    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect
    ax.set_facecolor('white')

    ax.set_xlim([-0.3, 0.3])
    ax.set_xticks([-0.2, 0.2])
    ax.set_ylim([-0.3, 0.3])
    ax.set_yticks([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.4])
    ax.set_zticks([-0.1, 0.3])
    ax.set_xlabel(r'$x~[m]$', fontsize=20)
    ax.set_ylabel(r'$y~[m]$', fontsize=20)
    ax.set_zlabel(r'$z~[m]$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)  # for x and y
    ax.tick_params(axis='z', which='major', labelsize=14)     # for z

    plt.tight_layout()
    plt.savefig(rf"{path_to_figures}/trajectory.svg")
        
        
def plot_trajectories_with_kettle(T, T_var, kettle, path_to_figures):

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Lighting setup
    ls = LightSource(azdeg=135, altdeg=45)

    N = T.shape[2]
    keyframes = [0, round(0.22* N ), round(0.42* N ), round(0.6* N ), N-1]
    
    # Calibrate object vertices
    vertices = np.vstack((kettle['homogeneous_vertices'][:3, :], kettle['homogeneous_vertices'][3, :]))
    vertices[:3, :] += np.array([[-0.05], [0.04], [-0.01]])  # Offset
    rot = R.from_euler('xzx', [180, 150, 4], degrees=True).as_matrix()
    T_obj = np.eye(4)
    T_obj[:3, :3] = rot
    vertices = T_obj @ vertices
        
    for k in keyframes:

        transformed_vertices = T[:, :, k] @ vertices

        # Face vertices for 3D collection
        poly3d = [[transformed_vertices[:3, j] for j in face] for face in kettle['faces']]
        verts_array = np.array([np.array(face) for face in poly3d])

        normals = compute_normals(verts_array)

        # Lighting shading via LightSource
        shade_vals = ls.shade_normals(normals, fraction=1.0)  # Grayscale
        cmap = plt.cm.Blues  # Or try 'viridis', 'plasma', etc.
        shade_colors = cmap(0.4/(shade_vals+0.01))  # Map grayscale to RGBA
        collection = Poly3DCollection(verts_array, facecolors=shade_colors, edgecolors='none', linewidths=0)
        collection.set_alpha(0.15)
        ax.add_collection3d(collection)

    # Plot trajectories of the origins of the different frames
    nb_lines = len(T_var)
    colors = ['b', 'r']

    for j in range(nb_lines):
        p = T_var[j][0:3, 3, :] 
        ax.plot(p[0, :], p[1, :], p[2, :], color=colors[j], linewidth=4)

        # Plot orientation body frame
        start_end_frames = [keyframes[0],keyframes[-1]]
        ROT = T_var[j][:3,:3,start_end_frames]

        arrow_len = 0.08  
        ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 0], ROT[1, 0], ROT[2, 0], length=arrow_len, normalize=True, color=colors[j], linewidth=3.0)
        ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 1], ROT[1, 1], ROT[2, 1], length=arrow_len, normalize=True, color=colors[j], linewidth=3.0)
        ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 2], ROT[1, 2], ROT[2, 2], length=arrow_len, normalize=True, color=colors[j], linewidth=3.0)

    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect
    ax.view_init(elev=30, azim=135)  # Better 3D angle
    ax.set_facecolor('white')
    ax.set_xlim([-0.55, 0.0])
    ax.set_xticks([-0.5,0.0])
    ax.set_ylim([1.95, 2.5])
    ax.set_yticks([2.0,2.4])
    ax.set_zlim([-2.2, -1.7])
    ax.set_zticks([-2.1,-1.8])
    ax.set_xlabel(r'$x~[m]$', fontsize=20)
    ax.set_ylabel(r'$y~[m]$', fontsize=20)
    ax.set_zlabel(r'$z~[m]$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)  # for x and y
    ax.tick_params(axis='z', which='major', labelsize=14)     # for z

    plt.tight_layout()
    plt.savefig(rf"{path_to_figures}/trajectory.svg")

def plot_twists(Xi, s_max, name, input_trajectory, path_to_figures,):

    colors = ['b','r']
    axes = ['x','y','z']

    fig = plt.figure(figsize=(4, 4.5))
    x_axis = np.linspace(0, s_max, Xi[0].shape[2])
    
    nb_rows = 3
    nb_columns = 2
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):

            plt.subplot(nb_rows,nb_columns, 1 + row_subplot*nb_columns + col_subplot)

            for j in range(2):
                plt.plot(x_axis, Xi[j][col_subplot*nb_rows + row_subplot,0,:], color=colors[j])

            if col_subplot < 0.5:
                plt.ylim([-4, 4])
                plt.yticks([-4, 0, 4])
                plt.title(rf'$\omega_{{{axes[row_subplot]}}}~[rad/s]$')
            else:
                plt.ylim([-1, 1])
                plt.yticks([-1, 0, 1])
                plt.title(rf'$v_{{{axes[row_subplot]}}}~[rad/s]$')

            if input_trajectory == 'pouring':
                plt.xlim([0, 2.8])
                plt.xticks([0, 0.7, 1.6, 2.8])
            else:
                plt.xlim([0, s_max])
                plt.xticks([0, s_max])

            plt.xlabel(r'$t~[s]$')
            plt.grid(True)

    fig.tight_layout()  # Adjust subplots to fit in the figure area.
    plt.savefig(rf'{path_to_figures}/{name}')

    
def plot_U(U, s_max, name, input_trajectory, path_to_figures):
    
    colors = ['b','r']
    linewidths = [3.0,1.5]

    fig = plt.figure(figsize=(6, 9))
    x_axis = np.linspace(0, s_max, U[0].shape[2])

    nb_rows = 6
    nb_columns = 3
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):

            plt.subplot(nb_rows,nb_columns, 1 + row_subplot*nb_columns + col_subplot)

            for j in range(2):
                plt.plot(x_axis, U[j][row_subplot,col_subplot,:], color=colors[j], linewidth = linewidths[j])

            # Annotate the plots
            eps_21 = (row_subplot == 1) and (col_subplot == 0)
            eps_31 = (row_subplot == 2) and (col_subplot == 0)
            eps_32 = (row_subplot == 2) and (col_subplot == 1)

            eps_51 = (row_subplot == 4) and (col_subplot == 0)
            eps_61 = (row_subplot == 5) and (col_subplot == 0)
            eps_62 = (row_subplot == 5) and (col_subplot == 1)

            residual_term = eps_21 or eps_31 or eps_32 or eps_51 or eps_61 or eps_62

            if residual_term:
                name_component = r'\epsilon'
            else:
                name_component = 'u'

            idx = f'{row_subplot+1}{col_subplot+1}'

            if row_subplot < 2.5:
                plt.ylim([-4, 4])
                plt.yticks([-4, 0, 4])
                plt.title(rf'${{{name_component}}}_{{{idx}}}~[rad/s]$')
            else:
                plt.ylim([-0.9, 0.9])
                plt.yticks([-0.9, 0, 0.9])
                plt.title(rf'${{{name_component}}}_{{{idx}}}~[m/s]$')

            if input_trajectory == 'pouring':
                plt.xlim([0, 2.8])
                plt.xticks([0, 0.7, 1.6, 2.8])
            else:
                plt.xlim([0, s_max])
                plt.xticks([0, s_max])

            plt.xlabel(r'$t~[s]$')
            plt.grid(True)

    fig.tight_layout()  # Adjust subplots to fit in the figure area.
    plt.savefig(rf'{path_to_figures}/{name}')




            
