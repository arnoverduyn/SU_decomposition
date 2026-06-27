# Import necessary libraries
import scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting
import matplotlib.pyplot as plt

############ Input ##########
input_trajectory = 'peg_on_hole_alignment'
# options: 'contour_following', peg_on_hole_alignment'
progress_domain = 'geometric'
# options: 'time', 'geometric'

############ Load and preprocess the trajectory and object data ##########
path_to_data = 'Data'
path_to_figures = 'figures'

# Load the trajectory data
T_raw, wrench_raw, N, dt, time_total = src.data_handling.load_demo_trajectory_force(input_trajectory,path_to_data)

if progress_domain == 'time':
    # Subsample raw trajectory data
    T, ds = T_raw[:,:,0:N:3], 3*dt
    wrench = wrench_raw[:,0:N:3]
    N = T.shape[2]
elif progress_domain == 'geometric':
    # Interpolate pose data to equidistant geometric progress steps
    s = src.robotics.calculate_geom_progress_axis(T_raw,dt, L=0.3)
    ds = 0.02 # -> 2 cm
    N = src.data_handling.calculate_number_of_equidistant_steps_in_array(s, stepsize = ds)
    s_equidistant = src.data_handling.make_array_equidistant(s, N)
    T = src.robotics.interpT(s, T_raw, s_equidistant)
    wrench = numpy.vstack([numpy.interp(s_equidistant, s, wrench_raw[i,:]) for i in range(wrench_raw.shape[0])])


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
wrench_var = [numpy.zeros(wrench.shape) for j in range(nb_body_frame_transformations)]

# Apply the body frame transformations
for j in range(nb_body_frame_transformations):
    for k in range(N):
        T_inv = src.robotics.inverse_T(body_frame_transformations[j])
        wrench_var[j][:3,k] = T_inv[:3,:3] @ wrench[:3,k] 
        wrench_var[j][3:6,k] = T_inv[:3,:3] @ wrench[3:6,k] + numpy.cross(T_inv[:3,3],wrench_var[j][:3,k])

############ Calculate the SU decomposition ########## 

# Initialise the results
Xi = [numpy.zeros((6,3,N-3)) for j in range(nb_body_frame_transformations)]
U = [numpy.zeros((6,3,N-3)) for j in range(nb_body_frame_transformations)]
U_reg = [numpy.zeros((6,3,N-3)) for j in range(nb_body_frame_transformations)]

for j in range(nb_body_frame_transformations):

    # Smooth the wrench trajectory
    wrench_smooth = scipy.ndimage.gaussian_filter1d(wrench_var[j], sigma = 1.0, axis=1, mode='nearest')

    # Perform the successive SU decompositions along the trajectory
    for k in range(N-3): 

        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi_ = numpy.column_stack([wrench_smooth[:,k], wrench_smooth[:,k+1], wrench_smooth[:,k+2]])

        # Compute U matrix without regularization
        U_, _, _ = src.SU_decomp.SU(Xi_)

        # Compute U matrix with regularization
        U_reg_, _, _ = src.SU_decomp.SU(Xi_, L = 0.3)

        # Store the results
        Xi[j][:,:,k] = Xi_
        U[j][:,:,k] = U_
        U_reg[j][:,:,k] = U_reg_


############ Plot the results ########## 
colors = ['r','b']
fig, axes = src.plotting.initialize_plot_wrench_trajectory(progress_domain, input_trajectory)
for j in range(nb_body_frame_transformations):
    axes = src.plotting.plot_twist_trajectory(axes, Xi[j][:,0,:], time_total, color = colors[j])
fig.savefig(rf"{path_to_figures}/wrenches.svg")

fig, axes = src.plotting.initialize_plot_U_wrench(progress_domain, input_trajectory)
linewidths = [3.0,1.5]
for j in range(nb_body_frame_transformations):
    axes = src.plotting.plot_U(axes, U[j], time_total, color = colors[j], linewidth = linewidths[j])
fig.savefig(rf"{path_to_figures}/U.svg")

fig, axes = src.plotting.initialize_plot_U(progress_domain, input_trajectory)
for j in range(nb_body_frame_transformations):
    axes = src.plotting.plot_U(axes, U_reg[j], time_total, color = colors[j], linewidth = linewidths[j])
fig.savefig(rf"{path_to_figures}/U_reg.svg")
