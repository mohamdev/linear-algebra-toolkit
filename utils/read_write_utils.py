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