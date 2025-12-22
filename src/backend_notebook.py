import scipy, numpy, src.data_handling, src.robotics, src.SU_decomp, src.plotting

def load_trajectory(input_trajectory,path_to_data):

    # Load the data
    T_raw, dt = src.data_handling.load_demo_trajectory(input_trajectory,path_to_data)
    N = T_raw.shape[2]

    # Subsample raw trajectory data
    T_sub = T_raw[:,:,0:N:3]
    dt = 3*dt

    return T_sub, dt

def compute_SU(T,dt,L):

    # Initialise the results
    N = T.shape[2]
    U = [numpy.zeros((6,3,N-3)), numpy.zeros((6,3,N-3))]

    # Calculate body twist trajectory
    twist = src.robotics.calculate_bodytwist_from_poses(T,dt)

    # Smooth the body twist trajectory
    twist_smooth = scipy.ndimage.gaussian_filter1d(twist, sigma = 1, axis=1, mode='nearest')

    # Perform the successive SU decompositions along the trajectory
    for k in range(N-3): 

        # Restructure twist data into successive overlapping windows of size (6,3)
        Xi_ = numpy.column_stack([twist_smooth[:,k], twist_smooth[:,k+1], twist_smooth[:,k+2]])

        # Compute U matrix without regularization
        U_, _, _ = src.SU_decomp.SU(Xi_)

        # Compute U matrix with regularization
        U_reg_, _, _ = src.SU_decomp.SU(Xi_, L = L)

        # Store the results
        U[0][:,:,k] = U_reg_
        U[1][:,:,k] = U_

    return U

def plot_trajectory(T,input_trajectory,path_to_data,path_to_figures):
    src.plotting.plot_trajectories(T, [T], input_trajectory, path_to_data, path_to_figures)

def plot_U(U,dt,input_trajectory,path_to_figures):
    N = U[0].shape[2]
    time_total = round((N-1)*dt,1)
    src.plotting.plot_U(U, time_total, 'U.svg', input_trajectory, path_to_figures)


