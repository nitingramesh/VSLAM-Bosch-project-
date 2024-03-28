import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
import sys

sys.path.append('..')
def read_txt_file(file_path):
    data_array = []

    try:
        with open(file_path, 'r') as file:
            # Skip lines until line 11
            for _ in range(10):
                next(file)

            # Read lines starting from line 11
            for line in file:
                # Split each line into an array of values
                values = line.strip().split(',')
                # Convert values to appropriate data types if needed
                # For example, you can use map(int, values) to convert all values to integers

                # Add the array of values to the 2D array
                data_array.append(values)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_array


def ply_to_txt(ply_file_path, txt_file_path):
    try:
        with open(ply_file_path, 'r') as ply_file:
            lines = ply_file.readlines()

            # Find the index where vertex data starts
            start_index = 11 #lines.index('end_header\n') + 1

            # Extract vertex data
            vertex_data = [line.split()[:3] for line in lines[start_index:]]

            # Convert vertex data to numpy array
            vertices = np.array(vertex_data, dtype=float)

            # Save the vertices to a text file
            np.savetxt(txt_file_path, vertices, delimiter=',', fmt='%.6f')

            print(f"Conversion successful. Text file saved at {txt_file_path}")

    except FileNotFoundError:
        print(f"File not found: {ply_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def depth_from_euclidean_distance(euclidean_distance, reference_values):
    """
    Map Euclidean distance to actual depth using linear interpolation.

    Parameters:
    - euclidean_distance: The Euclidean distance to be mapped.
    - reference_values: A list of tuples containing reference (distance, depth) pairs.

    Returns:
    - The estimated depth corresponding to the given Euclidean distance.
    """
    # Ensure at least two reference values are provided
    if len(reference_values) < 2:
        raise ValueError("At least two reference values are required.")

    # Sort reference values by distance
    reference_values = sorted(reference_values, key=lambda x: x[0])

    # Find the two closest reference values
    lower_ref, upper_ref = None, None
    for ref in reference_values:
        if ref[0] <= euclidean_distance:
            lower_ref = ref
        if ref[0] >= euclidean_distance:
            upper_ref = ref
            break

    # If only one reference value is found, return its corresponding depth
    if lower_ref is None or upper_ref is None:
        return lower_ref[1] if lower_ref is not None else upper_ref[1]

    # Perform linear interpolation
    lower_dist, lower_depth = lower_ref
    upper_dist, upper_depth = upper_ref

    # Avoid division by zero
    if lower_dist == upper_dist:
        return lower_depth

    # Calculate interpolated depth
    interpolated_depth = lower_depth + (euclidean_distance - lower_dist) * (
        (upper_depth - lower_depth) / (upper_dist - lower_dist)
    )

    return interpolated_depth

reference_values = [
    (15, 20),
    (23.8, 40),
    (34.28, 60),
    (48, 80),
    (60, 100)
]


# Example usage:
ply_file_path = '/home/user/Music/pointCloudDeepLearning3.ply'
txt_file_path = '/home/user/Music/pointCloudDeepLearning3.txt'
ply_to_txt(ply_file_path, txt_file_path)

# Example usage:
file_path = '/home/user/Music/pointCloudDeepLearning3.txt'
result = read_txt_file(file_path)
#print(result)

#print("load ply and stuff\n")

#pcd = o3d.io.read_point_cloud("/home/user/Music/pointCloudDeepLearning3.ply")

#Row major
point1 = [float(x) for x in result[240*640+320]]
point2 = [float(x) for x in result[425*640+320]]

#Column major
point3 = [float(x) for x in result[320*480+240]]
point4 = [float(x) for x in result[320*480+425]]

#result2 = [abs(x - y) for x, y in zip(list1_float, list2_float)]
#result2 = [abs(x - y) for x, y in zip(float(result[420+1+600]), float(result[320+240+1]))]


x1, y1, z1 = point1
x2, y2, z2 = point2

x3, y3, z3 = point3
x4, y4, z4 = point4

# Calculate Euclidean distance
distanceRow = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
distanceRow = distanceRow/1000
distanceCol = math.sqrt((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2)
distanceCol = distanceCol/1000

print("0 point coordinate: ", point1)
print("Center point coordinate: ", point2)
print("\nThe row euclidean distance is ", distanceRow)
#print("\nThe column euclidean distance is ", distanceCol)

actDistRow = depth_from_euclidean_distance(distanceRow, reference_values)
#actDistCol = depth_from_euclidean_distance(distanceCol, reference_values)
print("\nThe maybe real row euclidean distance is ", actDistRow)
#print("\nThe maybe real column euclidean distance is ", actDistCol)


#print("\n\n", pcd)
#print(np.asarray(pcd.points))

#o3d.visualization.draw_geometries([pcd])
