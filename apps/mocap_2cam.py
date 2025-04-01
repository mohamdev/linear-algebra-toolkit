#This script runs the calibration camera to mocap, and writes the transformation matrix it in a .txt
from utils.rigid_bodies_algorithms import *
from utils.read_write_utils import *
import pandas as pd

mks_names = ['B']
mks_features_names = []
for m in mks_names:
    mks_features_names = mks_features_names + [f"{m}x", f"{m}y", f"{m}z"]

position_names = ['tvec']
position = []
for p in position_names:
    position = position + [f"{p}_x",f"{p}_y",f"{p}_z"]

no_test = "rs"
mocap_df = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/barycenter_raw.csv", usecols=mks_features_names).values
postion_aruco = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/pose_aruco.csv", usecols=position).values
res_file = f"/root/workspace/ros_ws/src/rt-cosmik/output/test{no_test}/soder.txt"

mocap_df = np.array(mocap_df)
postion_aruco = np.array(postion_aruco)


print("-- Soder --")
R, d, rms = soder(postion_aruco,mocap_df) #mocap vers cam
print("R:\n", R)
print("d:\n", d)
print("rms:\n", rms)

save_transformation(res_file, R, d, 1.0, rms)

print("-- Challis --")
R, d, s, rms = challis(postion_aruco,mocap_df)
print("R:\n", R)
print("d:\n", d)
print("scale factor:\n", s)
print("rms:\n", rms)

# save_transformation("../data/challis.txt", R, d, 1.0, rms)
