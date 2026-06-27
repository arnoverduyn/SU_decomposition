import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation as R
import src.data_handling

def plot_trajectory_origin(ax, T, color = 'b', linewidth = 4.):
    p = T[0:3, 3, :]    
    ax.plot(p[0, :], p[1, :], p[2, :], color=color, linewidth=linewidth)
    return ax

def plot_frame(ax, T, color = 'b', linewidth = 3., arrow_len = 0.08):
    p = T[:3,3]
    R = T[:3,:3]
    for k in range(3):
        ax.quiver(p[0], p[1], p[2], R[0, k], R[1, k], R[2, k], length=arrow_len, normalize=True, color=color, linewidth=linewidth)
    return ax

def plot_frames(ax, T, key_values, color = 'b', linewidth = 3., arrow_len = 0.08):
    for j in key_values:
        ax = plot_frame(ax, T[:,:,j], color = color, linewidth = linewidth, arrow_len = arrow_len)
    return ax

def plot_rigid_body(ax, T, object_data):

    nb_vertices = object_data['vertices'].shape[0]
    hom_vertices = np.column_stack([object_data['vertices'],np.ones(nb_vertices)])
    transformed_vertices = T @ hom_vertices.T

    ls = LightSource(azdeg=135, altdeg=45)
    polygons = src.data_handling.construct_polygons(transformed_vertices[:3,:].T, object_data['faces'])
    normals = src.data_handling.compute_normals(polygons)
    shade_vals = ls.shade_normals(normals, fraction=1.0)  # Grayscale
    cmap = plt.cm.Blues  # Or try 'viridis', 'plasma', etc.
    shade_colors = cmap(0.4/(shade_vals+0.01))  # Map grayscale to RGBA
    collection = Poly3DCollection(polygons, facecolors=shade_colors, edgecolors='none', linewidths=0)
    collection.set_alpha(0.15)

    ax.add_collection3d(collection)
    return ax

def plot_rigid_bodies(ax, T, key_values, object_data):
    for j in key_values:
        ax = plot_rigid_body(ax, T[:,:,j], object_data)
    return ax

def initialize_plot_twist_trajectory(progress_domain = 'time', input_trajectory = 'pouring'):
    fig = plt.figure(figsize=(7, 3.5))
    axis_labels = ['x','y','z']

    if progress_domain == 'time':
        y_axis_units = ['rad/s', 'm/s']  
        progress_axis_units = 's'
    elif progress_domain == 'geometric':
        y_axis_units = ['rad/m', '-']  
        progress_axis_units = 'm'
    
    fontsize_axes = 13
    fontsize_title = 15
    axes = []
    nb_rows = 2
    nb_columns = 3
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):
            ax = fig.add_subplot(nb_rows, nb_columns, 1 + row_subplot*nb_columns + col_subplot)

            if row_subplot < 0.5:
                # ax.set_ylim([-2, 2])
                # ax.set_yticks([-2, 0, 2])
                ax.set_title(rf'$\omega_{{{axis_labels[col_subplot]}}}~[{{{y_axis_units[0]}}}]$',fontsize=fontsize_title)
            else:
                # ax.set_ylim([-0.5, 0.5])
                # ax.set_yticks([-0.5, 0, 0.5])
                ax.set_title(rf'$v_{{{axis_labels[col_subplot]}}}~[{{{y_axis_units[1]}}}]$',fontsize=fontsize_title)

            if input_trajectory == 'pouring' and progress_domain == 'time':
                ax.set_xlim([0, 5.6])
                ax.set_xticks([0, 1.4, 3.2, 5.6])

            ax.grid(True)

            # if col_subplot > 0.5:
            #    ax.tick_params(labelleft=False)
                
            if row_subplot > 0.5:
                ax.set_xlabel(rf'$s~[{{{progress_axis_units[0]}}}]$',fontsize=fontsize_axes)
            else:
                ax.tick_params(labelbottom=False)

            ax.tick_params(labelsize=fontsize_axes)
            axes.append(ax)

    fig.tight_layout()

    return fig, axes

def plot_twist_trajectory(axes, twist_trajectory, s_max, color = 'b'):

    progress_axis = np.linspace(0, s_max, twist_trajectory.shape[1])
    
    nb_rows = 2
    nb_columns = 3
    ax_counter = 0
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):

            ax = axes[ax_counter]
            ax.plot(progress_axis, twist_trajectory[row_subplot*nb_columns + col_subplot,:], color=color)
            ax_counter += 1

    return axes

def initialize_plot_wrench_trajectory(progress_domain = 'time', input_trajectory = 'pouring'):
    fig = plt.figure(figsize=(7, 3.5))
    axis_labels = ['x','y','z']

    y_axis_units = ['N', 'Nm']  
    if progress_domain == 'time':
        progress_axis_units = 's'
    elif progress_domain == 'geometric':
        progress_axis_units = 'm'
    
    fontsize_axes = 13
    fontsize_title = 15
    axes = []
    nb_rows = 2
    nb_columns = 3
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):
            ax = fig.add_subplot(nb_rows, nb_columns, 1 + row_subplot*nb_columns + col_subplot)

            if row_subplot < 0.5:
                # ax.set_ylim([-2, 2])
                # ax.set_yticks([-2, 0, 2])
                ax.set_title(rf'$f_{{{axis_labels[col_subplot]}}}~[{{{y_axis_units[0]}}}]$',fontsize=fontsize_title)
            else:
                # ax.set_ylim([-0.5, 0.5])
                # ax.set_yticks([-0.5, 0, 0.5])
                ax.set_title(rf'$\tau_{{{axis_labels[col_subplot]}}}~[{{{y_axis_units[1]}}}]$',fontsize=fontsize_title)

            if input_trajectory == 'pouring' and progress_domain == 'time':
                ax.set_xlim([0, 5.6])
                ax.set_xticks([0, 1.4, 3.2, 5.6])

            ax.grid(True)

            # if col_subplot > 0.5:
            #    ax.tick_params(labelleft=False)
                
            if row_subplot > 0.5:
                ax.set_xlabel(rf'$s~[{{{progress_axis_units[0]}}}]$',fontsize=fontsize_axes)
            else:
                ax.tick_params(labelbottom=False)

            ax.tick_params(labelsize=fontsize_axes)
            axes.append(ax)

    fig.tight_layout()

    return fig, axes


def initialize_plot_U(progress_domain = 'time', input_trajectory = 'pouring'):

    fig = plt.figure(figsize=(7, 9))
    axes = []

    if progress_domain == 'time':
        y_axis_units = ['rad/s', 'm/s']  
        progress_axis_units = 's'
    elif progress_domain == 'geometric':
        y_axis_units = ['rad/m', '-']  
        progress_axis_units = 'm'

    nb_rows = 6
    nb_columns = 3
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):

            ax = fig.add_subplot(nb_rows,nb_columns, 1 + row_subplot*nb_columns + col_subplot)

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

            fontsize_axes = 13
            fontsize_title = 15
            if row_subplot < 2.5:
                # ax.set_ylim([-2, 2])
                # ax.set_yticks([-2, 0, 2])
                ax.set_title(rf'${{{name_component}}}_{{{idx}}}~[{{{y_axis_units[0]}}}]$', fontsize=fontsize_title)
            else:
                # ax.set_ylim([-0.5, 0.5])
                # ax.set_yticks([-0.5, 0, 0.5])
                ax.set_title(rf'${{{name_component}}}_{{{idx}}}~[{{{y_axis_units[1]}}}]$', fontsize=fontsize_title)

            if input_trajectory == 'pouring' and progress_domain == 'time':
                ax.set_xlim([0, 5.6])
                ax.set_xticks([0, 1.4, 3.2, 5.6])

            ax.grid(True)

            # if col_subplot > 0.5:
            #    ax.tick_params(labelleft=False)

            if row_subplot > 4.5:
                ax.set_xlabel(rf'$s~[{{{progress_axis_units}}}]$', fontsize=fontsize_axes )
            else:
                ax.tick_params(labelbottom=False)

            ax.tick_params(labelsize=fontsize_axes)
            axes.append(ax)
            
    fig.tight_layout()
    return fig, axes

def initialize_plot_U_wrench(progress_domain = 'time', input_trajectory = 'pouring'):

    fig = plt.figure(figsize=(7, 9))
    axes = []

    y_axis_units = ['N', 'Nm']  
    if progress_domain == 'time':
        progress_axis_units = 's'
    elif progress_domain == 'geometric':
        progress_axis_units = 'm'

    nb_rows = 6
    nb_columns = 3
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):

            ax = fig.add_subplot(nb_rows,nb_columns, 1 + row_subplot*nb_columns + col_subplot)

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

            fontsize_axes = 13
            fontsize_title = 15
            if row_subplot < 2.5:
                # ax.set_ylim([-2, 2])
                # ax.set_yticks([-2, 0, 2])
                ax.set_title(rf'${{{name_component}}}_{{{idx}}}~[{{{y_axis_units[0]}}}]$', fontsize=fontsize_title)
            else:
                # ax.set_ylim([-0.5, 0.5])
                # ax.set_yticks([-0.5, 0, 0.5])
                ax.set_title(rf'${{{name_component}}}_{{{idx}}}~[{{{y_axis_units[1]}}}]$', fontsize=fontsize_title)

            if input_trajectory == 'pouring' and progress_domain == 'time':
                ax.set_xlim([0, 5.6])
                ax.set_xticks([0, 1.4, 3.2, 5.6])

            ax.grid(True)

            # if col_subplot > 0.5:
            #    ax.tick_params(labelleft=False)

            if row_subplot > 4.5:
                ax.set_xlabel(rf'$s~[{{{progress_axis_units}}}]$', fontsize=fontsize_axes )
            else:
                ax.tick_params(labelbottom=False)

            ax.tick_params(labelsize=fontsize_axes)
            axes.append(ax)
            
    fig.tight_layout()
    return fig, axes
    
def plot_U(axes, U, s_max, color = 'b', linewidth = 3.):
    
    progress_axis = np.linspace(0, s_max, U.shape[2])

    nb_rows = 6
    nb_columns = 3
    ax_counter = 0
    for row_subplot in range(nb_rows):
        for col_subplot in range(nb_columns):
            
            ax = axes[ax_counter]
            ax.plot(progress_axis, U[row_subplot,col_subplot,:], color=color, linewidth = linewidth)
            ax_counter += 1
            
    return axes

def ax_settings_pouring_trajectory(ax):
    ax.view_init(elev=30, azim=135)  # Better 3D angle
    ax.set_xlim([-0.55, 0.0])
    ax.set_xticks([-0.5,0.0])
    ax.set_ylim([1.95, 2.5])
    ax.set_yticks([2.0,2.4])
    ax.set_zlim([-2.2, -1.7])
    ax.set_zticks([-2.1,-1.8])
    return ax

def ax_settings_general(ax):
    ax.set_box_aspect([1, 1, 1])  # Equal aspect
    ax.tick_params(axis='both', which='major', labelsize=28)  # for x and y
    ax.tick_params(axis='z', which='major', labelsize=28)     # for z
    # ax.grid(False)
    return ax
            

