# Import necessary libraries
import sys, scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation as R__

path_to_data = 'Data'
path_to_figures = 'figures'

############ Load and preprocess the trajectory data ##########

############ Input ##########
input_trajectory = 'pouring' 
# options: 'helical_translation', 'axis_rotation', 'precession', 'pouring', 'contour_following',
#          'peg_on_hole_alignment'
progress_domain = 'geometric'
# options: 'time', 'geometric'


############ Load and preprocess the trajectory and object data ##########
path_to_data = 'Data'
path_to_figures = 'figures'

# Load the trajectory data
T_raw, N, dt, time_total = src.data_handling.load_demo_trajectory_motion(input_trajectory,path_to_data)

if progress_domain == 'time':
    # Subsample raw trajectory data
    T, ds = T_raw[:,:,0:N:3], 3*dt
    N = T.shape[2]
elif progress_domain == 'geometric':
    # Interpolate pose data to equidistant geometric progress steps
    s = src.robotics.calculate_geom_progress_axis(T_raw, dt, L=0.3)
    ds = 0.02 # -> 2 cm
    N = src.data_handling.calculate_number_of_equidistant_steps_in_array(s, stepsize = ds)
    s_equidistant = src.data_handling.make_array_equidistant(s, N)
    T = src.robotics.interpT(s, T_raw, s_equidistant)

# Load the data of the rigid body
if input_trajectory == 'pouring':
    object_data = src.data_handling.load_data_kettle(path_to_data)
    T_kettle_wrt_tracker = src.data_handling.load_tracker_kettle_calibration_data()
    nb_vertices = object_data['vertices'].shape[0]
    hom_vertices = numpy.column_stack([object_data['vertices'],numpy.ones(nb_vertices)])
    calibrated_vertices = T_kettle_wrt_tracker @ hom_vertices.T
    object_data['vertices'] = calibrated_vertices[:3,:].T
else:
    object_data = src.data_handling.create_cube_data()

# Plot the original rigid-body trajectory
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
key_values_body_frame, key_values_rigid_object = [0,-1], [0,-1]
ax = src.plotting.plot_trajectory_origin(ax, T, color = 'b', linewidth = 3.)
ax = src.plotting.plot_frames(ax, T, key_values_body_frame , color = 'b', linewidth = 3., arrow_len = 0.08)
ax = src.plotting.plot_rigid_bodies(ax, T, key_values_rigid_object, object_data)
ax = src.plotting.ax_settings_general(ax)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/input_trajectory.svg")

############ Calculate the SU decomposition ########## 

# Initialise the results
Xi = numpy.zeros((6,3,N-3))
U = numpy.zeros((6,3,N-3))

# Calculate body twist trajectory
twist = src.robotics.calculate_bodytwist_from_poses(T,ds)

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
fig, axes = src.plotting.initialize_plot_U(progress_domain, input_trajectory)
axes = src.plotting.plot_U(axes, U, time_total, color = 'b', linewidth = 2.0)
fig.savefig(rf"{path_to_figures}/U_reg.svg")

############ Test reconstruction ###########

# Initialization
T_rec = numpy.zeros((4,4,N))
T_rec[:,:,0:3] = T[:,:,0:3]
twist_rec = numpy.zeros((6,N-1))
twist_rec[:,0] = numpy.squeeze(src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,0:2],ds))
twist_rec[:,1] = numpy.squeeze(src.robotics.calculate_bodytwist_from_poses(T_rec[:,:,1:3],ds))

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
    T_rec[:,:,3+k] = T_rec[:,:,3+k-1] @ scipy.linalg.expm(twist_matrix*ds)

    if k < N-4:
        # Update twist window
        Xi_rec[:,0,k+1] = Xi_rec[:,1,k]
        Xi_rec[:,1,k+1] = Xi_rec[:,2,k]

MS_reconstruction_error = 0.
for k in range(N):
    error = numpy.sum((T_rec-T)**2)
    MS_reconstruction_error += error

# Plot the reconstructed rigid-body trajectory
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
key_values_body_frame, key_values_rigid_object = [0,-1], [0,-1]
ax = src.plotting.plot_trajectory_origin(ax, T_rec, color = 'r', linewidth = 3.)
ax = src.plotting.plot_frames(ax, T_rec, key_values_body_frame , color = 'r', linewidth = 3., arrow_len = 0.08)
ax = src.plotting.plot_rigid_bodies(ax, T_rec, key_values_rigid_object, object_data)
ax = src.plotting.ax_settings_general(ax)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/reconstructed_trajectory.svg")

##################### Trajectory generalization ####################################################
T_target = numpy.array([[0., 0., -1., -0.5],[1., 0., 0., 2.2],[0., -1., 0., -2.], [0.,  0.,  0.,  1.]])

nb_targets = 3
generated_trajectories = numpy.zeros((nb_targets,4,4,N))
for Q in range(nb_targets):

    T_target[0,3] += 0.25

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
        T_gen_next = T_gen[:,:,2+k] @ scipy.linalg.expm(twist_gen_matrix*ds)

        # Update trajectory
        T_gen[:,:,3+k] = correction_pose_matrix_in_world @ T_gen_next
        
    error_target = numpy.sum((T_gen[:,:,-1]-T_target)**2)

    generated_trajectories[Q,:,:,:] = T_gen

    print(error_target)


# Plot the reconstructed rigid-body trajectory
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
key_values_body_frame, key_values_rigid_object = [0,-1], [0,-1]
for Q in range(nb_targets):
    T_gen = generated_trajectories[Q,:,:,:]
    ax = src.plotting.plot_trajectory_origin(ax, T_gen, color = 'r', linewidth = 3.)
    ax = src.plotting.plot_frames(ax, T_gen, key_values_body_frame , color = 'r', linewidth = 3., arrow_len = 0.08)
    ax = src.plotting.plot_rigid_bodies(ax, T_gen, key_values_rigid_object, object_data)
    ax = src.plotting.ax_settings_general(ax)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/generated_trajectories.svg")
