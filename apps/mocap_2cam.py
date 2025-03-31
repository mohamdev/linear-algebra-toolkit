#This script runs the calibration camera to mocap, and writes the transformation matrix it in a .txt
from utils.rigid_bodies_algorithms import *
from utils.read_write_utils import *
import pandas as pd

def get_mks_array_from_dict(mks_dict):
    """
    Converts a dictionary of markers with 3D coordinates into a 2D numpy array.

    Parameters:
    mks_dict: dict
        Dictionary with marker names as keys and 3D coordinates (arrays) as values.

    Returns:
    mks_array: ndarray
        2D array where each row corresponds to the 3D coordinates of a marker.
    """
    n_mks = len(mks_dict)  # Number of markers
    mks_array = np.zeros((n_mks, 3))  # Initialize a 2D array for 3D coordinates
    for i, (key, coords) in enumerate(mks_dict.items()):
        mks_array[i] = coords  # Assign the 3D coordinates to the array
    return mks_array

mks_names = ['B']
mks_features_names = []
for m in mks_names:
    mks_features_names = mks_features_names + [f"{m}x", f"{m}y", f"{m}z"]

position_names = ['tvec']
position = []
for p in position_names:
    position = position + [f"{p}_x",f"{p}_y",f"{p}_z"]


no_test = 4
mocap_df = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/barycenter_raw.csv", usecols=mks_features_names).values
postion_aruco = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/pose_aruco.csv", usecols=position).values
# mocap_df = pd.read_csv("/root/workspace/ros_ws/src/rt-cosmik/output/aruco_pose_cam1.csv", usecols=position).values
# postion_aruco = pd.read_csv("/root/workspace/ros_ws/src/rt-cosmik/output/aruco_pose_cam2.csv", usecols=position).values
# n_imgs = 280
# mocap_df = mocap_df [:n_imgs]
# postion_aruco = postion_aruco [:n_imgs]
mocap_df = np.array(mocap_df)
postion_aruco = np.array(postion_aruco)

# print(mocap_df)
# mocap_mks_dict, _ = read_mks_data(mocap_df)
# lstm_mks_dict, _ = read_mks_data(postion_aruco)

# mks_mocap = get_mks_array_from_dict(mocap_mks_dict[0])
# mks_lstm = get_mks_array_from_dict(lstm_mks_dict[0])


print("-- Soder --")
R, d, rms = soder(mocap_df, postion_aruco) #mocap vers cam
print("R:\n", R)
print("d:\n", d)
print("rms:\n", rms)

save_transformation(f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/soder.txt", R, d, 1.0, rms)

print("-- Challis --")
R, d, s, rms = challis(mocap_df, postion_aruco)
print("R:\n", R)
print("d:\n", d)
print("scale factor:\n", s)
print("rms:\n", rms)

# save_transformation("../data/challis.txt", R, d, 1.0, rms)
