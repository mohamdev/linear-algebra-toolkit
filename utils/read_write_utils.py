import numpy as np

def read_mks_data(data_markers, start_sample=0):
    #the mks are ordered in a csv like this : "time,r.ASIS_study_x,r.ASIS_study_y,r.ASIS_study_z...."
    """    
    Parameters:
        data_markers (pd.DataFrame): The input DataFrame containing marker data.
        start_sample (int): The index of the sample to start processing from.
        
    Returns:
        list: A list of dictionaries where each dictionary contains markers with 3D coordinates.
        dict: A dictionary representing the markers and their 3D coordinates for the specified start_sample.
    """
    # Extract marker column names
    marker_columns = [col[:-2] for col in data_markers.columns if col.endswith("_x")]
    
    # Initialize the result list
    result_markers = []
    
    # Iterate over each row in the DataFrame
    for _, row in data_markers.iterrows():
        frame_dict = {}
        for marker in marker_columns:
            x = row[f"{marker}_x"]
            y = row[f"{marker}_y"]
            z = row[f"{marker}_z"]
            frame_dict[marker] = np.array([x, y, z])  # Store as a NumPy array
        result_markers.append(frame_dict)
    
    # Get the data for the specified start_sample
    lstm_dict = result_markers[start_sample]
    
    return result_markers, lstm_dict

def save_transformation(file_path, R, d, s, rms):
    """
    Saves the transformation parameters (R, d, s, rms) to a text file.

    Parameters:
    file_path: str
        Path to the file where the transformation parameters will be saved.
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'w') as f:
        f.write("# Transformation Parameters\n")
        f.write("Rotation Matrix (R):\n")
        np.savetxt(f, R, fmt='%0.6f')
        f.write("Translation Vector (d):\n")
        np.savetxt(f, d.reshape(1, -1), fmt='%0.6f')
        f.write(f"Scale Factor (s): {s:.6f}\n")
        f.write(f"RMS Error: {rms:.6f}\n")

def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start:R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start:d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms

# Example usage (optional):
# R, d, s, rms = np.eye(3), np.array([1, 2, 3]), 1.0, 0.1
# save_transformation("transformation.txt", R, d, s, rms)
# R_loaded, d_loaded, s_loaded, rms_loaded = load_transformation("transformation.txt")