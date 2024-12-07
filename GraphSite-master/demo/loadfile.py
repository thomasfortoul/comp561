import numpy as np

# Load the .npy file
filename = "6ymw_B_dismap.npy"  # Replace with your file name
distance_matrix = np.load(filename)

# Print the loaded matrix
print("Loaded Distance Matrix:")
print(distance_matrix)