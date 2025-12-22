# Import necessary libraries
import sys, scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting
path_to_data = 'Data'
path_to_figures = 'figures'

############ Load and preprocess the trajectory data ##########

# Load the data
input_trajectory = 'helical_translation'  # options: 'helical_translation', 'axis_rotation', 'precession', 'pouring'
T_raw, dt = src.data_handling.load_demo_trajectory(input_trajectory,path_to_data)
N = T_raw.shape[2]
time_total = (N-1)*dt

# Subsample raw trajectory data
T_sub = T_raw[:,:,0:N:3]
dt = 3*dt
N = T_sub.shape[2]


############ Introduce variations in coordinate frame ##########

# Initialize the transformation matrices of the body frame
body_frame_transformations = [numpy.eye(4), numpy.eye(4)]

# Calibrate a body frame near handle
body_frame_transformations[0][:3,3] = numpy.array([0.07,-0.085,0.02]) 

# Calibrate a body frame near spout
rot2 = scipy.spatial.transform.Rotation.from_euler('xzx', [120, 70, 0], degrees=True).as_matrix()
body_frame_transformations[1][:3,:3] = rot2
body_frame_transformations[1][:3,3] = numpy.array([0.06,0.07,-0.015]) 

# Initialise the resulting trajectories
T_var = [numpy.zeros(T_sub.shape), numpy.zeros(T_sub.shape)]

# Apply the body frame transformations
for j in range(2):
    for k in range(N):
        T_var[j][:,:,k] = T_sub[:,:,k] @ body_frame_transformations[j]


############ Calculate the SU decomposition ########## 

# Initialise the results
Xi = [numpy.zeros((6,3,N-3)), numpy.zeros((6,3,N-3))]
U = [numpy.zeros((6,3,N-3)), numpy.zeros((6,3,N-3))]
U_reg = [numpy.zeros((6,3,N-3)), numpy.zeros((6,3,N-3))]

for j in range(2):

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
        U_reg_, _, _ = src.SU_decomp.SU(Xi_, L = 0.0)

        # Store the results
        Xi[j][:,:,k] = Xi_
        U[j][:,:,k] = U_
        U_reg[j][:,:,k] = U_reg_


############ Plot the results ########## 
(kettle,_,_,_) = src.data_handling.retrieve_data_designed_objects(path_to_data)
src.plotting.plot_trajectories(T_sub, T_var, kettle, input_trajectory, path_to_figures)
src.plotting.plot_twists(Xi, time_total, 'twists.svg', input_trajectory, path_to_figures)
src.plotting.plot_U(U, time_total, 'U.svg', input_trajectory, path_to_figures)
src.plotting.plot_U(U_reg, time_total, 'U_reg.svg', input_trajectory, path_to_figures)