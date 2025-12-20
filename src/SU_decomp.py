import numpy as np
from src.robotics import skew

def RU_signed(A):
    """
    QR-decomposition using Gramm-Shmidt with sign conventions for Q in SO3
    
    Input:
    A: np.ndarray of shape (3, 3)

    Output:
    R: np.ndarray of shape (3, 3)
    U: np.ndarray of shape (3, 3)
    """
    
    # Gramm-Shmidt orthogonalisation
    ex = A[:, 0] / np.linalg.norm(A[:, 0])
    
    proj = np.dot(A[:, 1], ex) * ex
    ey = A[:, 1] - proj
    ey = ey / np.linalg.norm(ey)
    
    ez = np.cross(ex, ey)
    R = np.column_stack((ex, ey, ez))

    U = R.T @ A

    return R, U


def SU(X, L=10**10):
    """
    Python equivalent of MATLAB eQR function.
    
    Parameters:
    X: np.ndarray of shape (6, 3)
    regularization: bool
    L: float

    Returns:
    R: np.ndarray of shape (3, 3)
    p: np.ndarray of shape (3,)
    U: np.ndarray of shape (6, 3)
    """
    
    # Calculate R
    X1 = X[0:3, :]
    X2 = X[3:6, :]
    R, U1 = RU_signed(X1)
    RTX2 = R.T @ X2

    # Calculate p_star
    z = RTX2[1, 0] / U1[0, 0]
    y = -RTX2[2, 0] / U1[0, 0]
    x = (RTX2[2, 1] + U1[0, 1] * y) / U1[1, 1]
    p_star = np.array([x, y, z])
    
    # Perform regularization
    if x**2 + y**2 + z**2 > L**2: # regularization active

        # Regularize p_star
        if y**2 + z**2 > L**2:  # Case 2
            x = 0
            y = L*y/np.sqrt(y**2 + z**2)
            z = L*z/np.sqrt(y**2 + z**2)
        else: # Case 1
            x = np.sign(x)*np.sqrt(L**2 - y**2 - z**2)
        p_star = np.array([x,y,z])
        
        p = R @ p_star
        U2 = RTX2 - skew(p_star) @ U1

        # Regularize R
        _ , U2_tri = RU_signed(U2)

        # input matrix = [L*U1 U2]
        # target matrix = [L*U1 U2_tri]
        C = L**2 * U1 @ U1.T + U2 @ U2_tri.T
        
        # Compute SVD
        U, _, Vt = np.linalg.svd(C)
        V = Vt.T
        Rc = V @ U.T
        
        # Ensure Rc is a proper rotation matrix (det = +1)
        if np.linalg.det(Rc) < 0:
            U[:, -1] *= -1
            Rc = V @ U.T
        U1 = Rc @ U1
        U2 = Rc @ U2
        R = R @ Rc.T
    else:
        p = R @ p_star
        U2 = RTX2 - skew(p_star) @ U1

    U = np.vstack((U1, U2))
    
    return U, R, p