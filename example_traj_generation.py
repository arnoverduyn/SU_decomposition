# Import necessary libraries
import sys, scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation as R__

path_to_data = 'Data'
path_to_figures = 'figures'

############ Load and preprocess the trajectory data ##########

# Load the data
input_trajectory = 'pouring'  # options: 'helical_translation', 'axis_rotation', 'precession', 'pouring'
T_raw, dt = src.data_handling.load_demo_trajectory(input_trajectory,path_to_data)
N = T_raw.shape[2]
time_total = (N-1)*dt

# Subsample raw trajectory data
T_sub = T_raw[:,:,0:N:3]
dt = 3*dt
N = T_sub.shape[2]

############ Calculate the SU decomposition ########## 

# Initialise the results
Xi = numpy.zeros((6,3,N-3))
U = numpy.zeros((6,3,N-3))

# Calculate body twist trajectory
twist = src.robotics.calculate_bodytwist_from_poses(T_sub,dt)

# Perform the successive SU decompositions along the trajectory
for k in range(N-3): 

    # Restructure twist data into successive overlapping windows of size (6,3)
    Xi_ = numpy.column_stack([twist[:,k], twist[:,k+1], twist[:,k+2]])

    # Compute U matrix with regularization
    U_, _, _ = src.SU_decomp.SU(Xi_, L = 0.3)

    # Store the results
    Xi[:,:,k] = Xi_
    U[:,:,k] = U_


############ Plot the results ########## 
nb_traj = 1
src.plotting.plot_trajectories(T_sub, T_sub, nb_traj, input_trajectory, path_to_data, path_to_figures, name = "trajectory.svg")
src.plotting.plot_U(U, time_total, nb_traj, 'U.svg', input_trajectory, path_to_figures)

############ Test reconstruction ###########

# Initialization
T_rec = numpy.zeros((4,4,N))
T_rec[:,:,0:3] = T_sub[:,:,0:3]
twist_rec = numpy.zeros((6,N-1))
twist_rec[:,0] = numpy.squeeze(src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,0:2],dt))
twist_rec[:,1] = numpy.squeeze(src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,1:3],dt))

Xi_rec = numpy.zeros((6,3,N-3))
Xi_rec[:,0,0] = twist_rec[:,0]
Xi_rec[:,1,0] = twist_rec[:,1]

twist_matrix = numpy.zeros((4,4))
twist_ = numpy.zeros(6)

# Reconstruction by integration
for k in range(N-3):

    # Reconstruct moving frame
    _, R, p = src.SU_decomp.SU(Xi_rec[:,:,k], L = 0.3)
    
    # Reconstruct twist
    twist_[0:3] = R @ U[0:3,2,k]
    twist_[3:6] = R @ U[3:6,2,k] - numpy.cross(twist_[0:3], p)
    twist_rec[:,2+k] = twist_
    Xi_rec[:,2,k] = twist_

    # Reconstruct pose
    twist_matrix[0:3,0:3] = src.robotics.skew(twist_[0:3])
    twist_matrix[0:3,3] = twist_[3:6]
    T_rec[:,:,3+k] = T_rec[:,:,3+k-1] @ scipy.linalg.expm(twist_matrix*dt)

    if k < N-4:
        # Update twist window
        Xi_rec[:,0,k+1] = Xi_rec[:,1,k]
        Xi_rec[:,1,k+1] = Xi_rec[:,2,k]

src.plotting.plot_trajectories(T_rec, T_rec, nb_traj, input_trajectory, path_to_data, path_to_figures, name = "reconstruction.svg")

##################### Trajectory generalization ####################################################
# T_target = numpy.array([[0., 0., -1., 0.5],[1., 0., 0., 2.],[0., -1., 0., -2.], [0.,  0.,  0.,  1.]])
T_target = numpy.array([[0., 0., -1., -0.5],[1., 0., 0., 2.2],[0., -1., 0., -2.], [0.,  0.,  0.,  1.]])

nb_targets = 5
generated_trajectories = numpy.zeros((nb_targets,4,4,N))
for Q in range(nb_targets):

    T_target[0,3] += 0.15

    error_target = numpy.sum((T_rec[:,:,-1]-T_target)**2)

    # Calculate pose difference to target pose
    finite_twist_matrix_final_body = src.robotics.logm_pose(src.robotics.inverse_T(T_rec[:,:,-1]) @ T_target)
    correction_pose_matrix_final_body = src.robotics.expm(finite_twist_matrix_final_body/(N-3))
    correction_pose_matrix_in_world = T_rec[:,:,-1] @ correction_pose_matrix_final_body @ src.robotics.inverse_T(T_rec[:,:,-1])

    # Initialise generated trajectory
    T_gen = numpy.zeros((4,4,N))
    T_gen[:,:,0:3] = T_rec[:,:,0:3]
    twist_gen = twist_rec
    twist_gen_matrix = numpy.zeros((4,4))

    for k in range(N-3):
        
        twist_gen_matrix[:3,:3] = src.robotics.skew(twist_gen[:3,k+2])
        twist_gen_matrix[:3,3] = twist_gen[3:6,k+2]
        T_gen_next = T_gen[:,:,2+k] @ scipy.linalg.expm(twist_gen_matrix*dt)

        # Update trajectory
        T_gen[:,:,3+k] = correction_pose_matrix_in_world @ T_gen_next
        
    error_target = numpy.sum((T_gen[:,:,-1]-T_target)**2)

    generated_trajectories[Q,:,:,:] = T_gen

    print(error_target)


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')

# Lighting setup
ls = LightSource(azdeg=135, altdeg=45)
keyframes = [0, N-1]

# Calibrate object vertices
(kettle,_,_,_) = src.data_handling.retrieve_data_designed_objects(path_to_data)
vertices = numpy.vstack((kettle['homogeneous_vertices'][:3, :], kettle['homogeneous_vertices'][3, :]))
vertices[:3, :] += numpy.array([[-0.05], [0.04], [-0.01]])  # Offset
rot = R__.from_euler('xzx', [180, 150, 4], degrees=True).as_matrix()
T_obj = numpy.eye(4)
T_obj[:3, :3] = rot
vertices = T_obj @ vertices
    
for Q in range(nb_targets):
    T = generated_trajectories[Q,:,:,:]

    for k in keyframes:

        transformed_vertices = T[:, :, k] @ vertices

        # Face vertices for 3D collection
        poly3d = [[transformed_vertices[:3, j] for j in face] for face in kettle['faces']]
        verts_array = numpy.array([numpy.array(face) for face in poly3d])

        normals = src.plotting.compute_normals(verts_array)

        # Lighting shading via LightSource
        shade_vals = ls.shade_normals(normals, fraction=1.0)  # Grayscale
        cmap = plt.cm.Blues  # Or try 'viridis', 'plasma', etc.
        shade_colors = cmap(0.4/(shade_vals+0.01))  # Map grayscale to RGBA
        collection = Poly3DCollection(verts_array, facecolors=shade_colors, edgecolors='none', linewidths=0)
        collection.set_alpha(0.15)
        ax.add_collection3d(collection)

    # Plot trajectories of the origin of the body
    start_end_frames = [keyframes[0],keyframes[-1]]
    p = T[0:3, 3, :] 

    # Plot continuous trajectory of frame origin    
    ax.plot(p[0, :], p[1, :], p[2, :], color = 'b', linewidth=4)

    # Plot body frames at start and end
    ROT = T[:3,:3,start_end_frames]
    arrow_len = 0.08  
    ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 0], ROT[1, 0], ROT[2, 0], length=arrow_len, normalize=True, color = 'b', linewidth=3.0)
    ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 1], ROT[1, 1], ROT[2, 1], length=arrow_len, normalize=True, color = 'b', linewidth=3.0)
    ax.quiver(p[0, start_end_frames], p[1, start_end_frames], p[2, start_end_frames], ROT[0, 2], ROT[1, 2], ROT[2, 2], length=arrow_len, normalize=True, color = 'b', linewidth=3.0)

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
ax.set_xlabel(r'$x~[m]$', fontsize=30)
ax.set_ylabel(r'$y~[m]$', fontsize=30)
ax.set_zlabel(r'$z~[m]$', fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=20)  # for x and y
ax.tick_params(axis='z', which='major', labelsize=20)     # for z
ax.set_axis_off()

plt.tight_layout()
plt.savefig(rf"{path_to_figures}/generation_multiple.svg")