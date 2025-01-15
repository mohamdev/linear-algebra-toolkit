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

#mks used for the mocap 2 cam calibration
mks_names = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
                'r_mknee_study','r_ankle_study','r_mankle_study', 
                'L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
                'r_shoulder_study','L_shoulder_study',
                'C7_study', 'r_lelbow_study',
                'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
                'L_lwrist_study','L_mwrist_study']

mks_features_names = []
for m in mks_names:
    mks_features_names = mks_features_names + [f"{m}_x", f"{m}_y", f"{m}_z"]

print(mks_features_names)
mocap_df = pd.read_csv("../data/mocap_mean.csv", usecols=mks_features_names)
lstm_df = pd.read_csv("../data/lstm_mean.csv", usecols=mks_features_names)

mocap_mks_dict, _ = read_mks_data(mocap_df)
lstm_mks_dict, _ = read_mks_data(lstm_df)

mks_mocap = get_mks_array_from_dict(mocap_mks_dict[0])
mks_lstm = get_mks_array_from_dict(lstm_mks_dict[0])

print("-- Soder --")
R, d, rms = soder(mks_mocap, mks_lstm)
print("R:\n", R)
print("d:\n", d)
print("rms:\n", rms)

print("-- Challis --")
R, d, s, rms = challis(mks_mocap, mks_lstm)
print("R:\n", R)
print("d:\n", d)
print("scale factor:\n", s)
print("rms:\n", rms)

# print(np.shape(mocap_mks_dict[0]['r.ASIS_study']))