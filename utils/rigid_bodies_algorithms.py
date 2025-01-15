import numpy as np

def challis(x, y):
    """
    Function to determine rigid body rotation, translation, scale, and RMS fit error.

    Parameters:
    x: ndarray
        3-D marker coordinates in position 1 (shape: n_markers x 3)
    y: ndarray
        3-D marker coordinates in position 2 (shape: n_markers x 3)

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error of the rigid body model

    The rigid body model is: y = s * R * x + d
    """
    # Example usage (optional):
    # x = np.array([[...], [...], ...])  # Replace with actual marker coordinates
    # y = np.array([[...], [...], ...])
    # R, d, s, rms = challis(x, y)

    # Number of markers and dimensions
    nmarkers, ndimensions = x.shape

    if ndimensions != 3:
        raise ValueError("Input marker coordinates must have 3 columns for 3D data.")

    # Compute means
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)


    # Construct matrices A and B, subtracting the mean
    A = (x - mx).T  # Shape: 3 x n_markers
    B = (y - my).T  # Shape: 3 x n_markers

    # Compute the cross-dispersion matrix
    C = (B @ A.T) / nmarkers  # 1/n (Challis)

    # Perform Singular Value Decomposition (SVD)
    P, _, Q = np.linalg.svd(C)

    # Calculate rotation matrix R with det(R) = 1
    R = P @ np.diag([1, 1, np.linalg.det(P @ Q.T)]) @ Q.T

    # Compute scale factor (Challis)
    sigmax2 = np.sum(A**2) / nmarkers
    s = (1 / sigmax2) * np.trace(R.T @ C)

    # Calculate the translation vector from the centroid of all markers
    d = my - s * R @ mx

    # Calculate RMS value of residuals
    sumsq = 0
    for i in range(nmarkers):
        y_pred = s * R @ x[i, :] + d
        sumsq += np.linalg.norm(y_pred - y[i, :])**2

    rms = np.sqrt(sumsq / (3 * nmarkers))

    return R, d, s, rms

# Unit test for the challis function
def test_challis():
    """Unit test for the challis function using known transformation parameters."""
    # Known transformation parameters
    R_known = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])  
    d_known = np.array([0.0, 5.0, 0.0])  # Translation vector
    s_known = 1.0  # Scale factor

    # Generate synthetic data
    x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    
    y = (s_known * (R_known @ x.T).T) + d_known  # Apply known transformation

    # Call the challis function
    R, d, s, rms = challis(x, y)
    print(R)
    print(d)
    print(s)

    # Assertions to check if the estimated parameters match the known ones
    assert np.allclose(R, R_known, atol=1e-3), f"Rotation matrix mismatch: {R} vs {R_known}"
    assert np.allclose(d, d_known, atol=1e-3), f"Translation vector mismatch: {d} vs {d_known}"
    assert np.isclose(s, s_known, atol=1e-3), f"Scale factor mismatch: {s} vs {s_known}"
    print(f"Test passed! RMS error: {rms:.6f}")

# Run the unit test
# test_challis()



import numpy as np

def soder(x, y):
    """
    Function to determine rigid body rotation, translation, and RMS fit error.

    Parameters:
    x: ndarray
        3-D marker coordinates in position 1 (shape: n_markers x 3)
    y: ndarray
        3-D marker coordinates in position 2 (shape: n_markers x 3)

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    rms: float
        Root mean square fit error of the rigid body model

    The rigid body model is: y = R * x + d
    """
    # Number of markers and dimensions
    nmarkers, ndimensions = x.shape

    if ndimensions != 3:
        raise ValueError("Input marker coordinates must have 3 columns for 3D data.")

    # Compute means
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)

    # Construct matrices A and B, subtracting the mean
    A = (x - mx).T  # Shape: 3 x n_markers
    B = (y - my).T  # Shape: 3 x n_markers

    # Compute the cross-dispersion matrix
    C = B @ A.T

    # Perform Singular Value Decomposition (SVD)
    P, _, Q = np.linalg.svd(C)

    # Calculate rotation matrix R with det(R) = 1
    R = P @ np.diag([1, 1, np.linalg.det(P @ Q.T)]) @ Q.T

    # Calculate the translation vector from the centroid of all markers
    d = my - R @ mx

    # Calculate RMS value of residuals
    sumsq = 0
    for i in range(nmarkers):
        y_pred = R @ x[i, :] + d
        sumsq += np.linalg.norm(y_pred - y[i, :])**2

    rms = np.sqrt(sumsq / (3 * nmarkers))

    return R, d, rms

# Example usage (optional):
# x = np.array([[...], [...], ...])  # Replace with actual marker coordinates
# y = np.array([[...], [...], ...])
# R, d, rms = soder(x, y)

# Unit test for the challis function
def test_soder():
    """Unit test for the soder function using known transformation parameters."""
    # Known transformation parameters
    R_known = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])  
    d_known = np.array([10.0, 0.0, 0.0])  # Translation vector

    x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

    y =  (R_known @ x.T).T + d_known  # Apply known transformation

    print("x:", x)
    print("y:", y)
    
    # Call the challis function
    R, d, rms = soder(x, y)
    print("R:\n", R)
    print("d:\n", d)
    print("rms:\n", rms)

    # Assertions to check if the estimated parameters match the known ones
    # assert np.allclose(R, R_known, atol=1e-3), f"Rotation matrix mismatch: {R} vs {R_known}"
    # assert np.allclose(d, d_known, atol=1e-3), f"Translation vector mismatch: {d} vs {d_known}"
    # assert np.isclose(s, s_known, atol=1e-3), f"Scale factor mismatch: {s} vs {s_known}"
    # print(f"Test passed! RMS error: {rms:.6f}")

