# Import necessary libraries
import sys, src.backend_notebook

# Add Python paths
sys.path.append('Data/')

# User inputs (free to change)
input_trajectory = 'pouring'  # options: 'pouring'
L = 0.3 # expressed in meters

# Load the trajectory data
T, dt = src.backend_notebook.load_trajectory(input_trajectory)

# Plot the input trajectory
src.backend_notebook.plot_trajectory(T)

# Calculate U matrix (both not regularized and regularized)
U = src.backend_notebook.compute_SU(T,dt,L)

src.backend_notebook.plot_U(U,dt)

