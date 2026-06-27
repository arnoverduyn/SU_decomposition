# Import necessary libraries
import scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting
import matplotlib.pyplot as plt

############ Input ##########
input_trajectory = 'precession'  # options: 'helical_translation', 'axis_rotation', 'precession', 'pouring'

############ Load and preprocess the trajectory and object data ##########
path_to_data = 'Data'
path_to_figures = 'figures'

# Load the trajectory data
T_raw, N, dt, time_total = src.data_handling.load_demo_trajectory(input_trajectory,path_to_data)

# Subsample raw trajectory data
T_sub, dt = T_raw[:,:,0:N:3], 3*dt
N = T_sub.shape[2]

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
ax = src.plotting.plot_trajectory_origin(ax, T_sub, color = 'b', linewidth = 3.)
ax = src.plotting.plot_frames(ax, T_sub, key_values_body_frame , color = 'b', linewidth = 3., arrow_len = 0.08)
ax = src.plotting.plot_rigid_bodies(ax, T_sub, key_values_rigid_object, object_data)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/input_trajectory.svg")

############ Introduce variations in coordinate frame ##########
nb_body_frame_transformations = 2

# Initialize the transformation matrices of the body frame
body_frame_transformations = [numpy.eye(4) for j in range(nb_body_frame_transformations)]

# Define body frame 1
body_frame_transformations[0][:3,3] = numpy.array([0.1,-0.13,0.04]) 

# Define body frame 2
body_frame_transformations[1][:3,3] = numpy.array([0.1,0.08,-0.04]) 
rot2 = scipy.spatial.transform.Rotation.from_euler('xzx', [120, 70, 0], degrees=True).as_matrix()
body_frame_transformations[1][:3,:3] = rot2

# Initialise the resulting trajectories
T_var = [numpy.zeros(T_sub.shape) for j in range(nb_body_frame_transformations)]

# Apply the body frame transformations
for j in range(nb_body_frame_transformations):
    for k in range(N):
        T_var[j][:,:,k] = T_sub[:,:,k] @ body_frame_transformations[j]

# Plot the rigid-body trajectories with new body frames
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax = src.plotting.plot_rigid_bodies(ax, T_sub, key_values_rigid_object, object_data)
key_values_body_frame, key_values_rigid_object = [0,-1], [0,-1]
colors = ['r','b']
for j in range(nb_body_frame_transformations):
    ax = src.plotting.plot_trajectory_origin(ax, T_var[j], color = colors[j], linewidth = 3.)
    ax = src.plotting.plot_frames(ax, T_var[j], key_values_body_frame , color = colors[j], linewidth = 3., arrow_len = 0.08)
if input_trajectory == 'pouring':
    ax = src.plotting.ax_settings_pouring_trajectory(ax)
fig.savefig(rf"{path_to_figures}/trajectories_with_different_body_frames.svg")

############ Calculate the SU decomposition ########## 

# Initialise the results
Xi = [numpy.zeros((6,3,N-3)) for j in range(nb_body_frame_transformations)]
U = [numpy.zeros((6,3,N-3)) for j in range(nb_body_frame_transformations)]
U_reg = [numpy.zeros((6,3,N-3)) for j in range(nb_body_frame_transformations)]

for j in range(nb_body_frame_transformations):

    # Calculate body twist trajectory
    twist = src.robotics.calculate_bodytwist_from_poses(T_var[j],dt)

    # Smooth the body twist trajectory
    twist_smooth = scipy.ndimage.gaussian_filter1d(twist, sigma= 1, axis=1, mode='nearest')

    # Perform the successive SU decompositions along the trajectory
    for k in range(N-3): 

        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi_ = numpy.column_stack([twist_smooth[:,k], twist_smooth[:,k+1], twist_smooth[:,k+2]])

        # Compute U matrix without regularization
        U_, _, _ = src.SU_decomp.SU(Xi_)

        # Compute U matrix with regularization
        U_reg_, _, _ = src.SU_decomp.SU(Xi_, L = 0.3)

        # Store the results
        Xi[j][:,:,k] = Xi_
        U[j][:,:,k] = U_
        U_reg[j][:,:,k] = U_reg_


############ Plot the results ########## 
fig, axes = src.plotting.initialize_plot_twist_trajectory(input_trajectory == 'pouring')
for j in range(nb_body_frame_transformations):
    axes = src.plotting.plot_twist_trajectory(axes, Xi[j][:,0,:], time_total, color = colors[j])
fig.savefig(rf"{path_to_figures}/twists.svg")

fig, axes = src.plotting.initialize_plot_U(input_trajectory == 'pouring')
linewidths = [3.0,1.5]
for j in range(nb_body_frame_transformations):
    axes = src.plotting.plot_U(axes, U[j], time_total, color = colors[j], linewidth = linewidths[j])
fig.savefig(rf"{path_to_figures}/U.svg")

fig, axes = src.plotting.initialize_plot_U(input_trajectory == 'pouring')
for j in range(nb_body_frame_transformations):
    axes = src.plotting.plot_U(axes, U_reg[j], time_total, color = colors[j], linewidth = linewidths[j])
fig.savefig(rf"{path_to_figures}/U_reg.svg")
