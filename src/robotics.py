import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

def skew(v):
    """Return the skew-symmetric matrix of a 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def extract_omega(R1, R2, ds):
    omega_skew = logm_rot(R2 @ R1.T) / ds
    return extract_vector_from_skew(omega_skew)

def extract_vector_from_skew(skew_matrix):
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]])

def hamilton_product(q, r):
    if len(q) != 4 or len(r) != 4:
        raise ValueError('Both inputs must be 4-element vectors.')
    
    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r
    
    return np.array([
        q0 * r0 - q1 * r1 - q2 * r2 - q3 * r3,
        q0 * r1 + q1 * r0 + q2 * r3 - q3 * r2,
        q0 * r2 - q1 * r3 + q2 * r0 + q3 * r1,
        q0 * r3 + q1 * r2 - q2 * r1 + q3 * r0
    ])

def interpT(x, T, xq):
    if xq[0] < x[0] or xq[-1] > x[-1]:
        raise ValueError('Cannot interpolate beyond first or last sample!')

    M = len(xq)
    T_interpolated = np.zeros((4, 4, M))

    j = 0
    for i in range(M):
        while xq[i] > x[j + 1]:
            j += 1

        x0, x1 = x[j], x[j + 1]
        T0, T1 = T[:, :, j], T[:, :, j + 1]

        if x1 - x0 != 0:
            T_new = T0 @ (expm((xq[i] - x0) / (x1 - x0) * logm_pose(inverse_T(T0) @ T1)))
        else:
            T_new = T0

        T_interpolated[:, :, i] = T_new

    return T_interpolated

def inverse_T(T):
    
    Tinv = np.copy(T)
    
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    p_new = -R.T @ p
    Tinv[0:4, :] = np.vstack((np.hstack((R.T, p_new[:, np.newaxis])), np.array([0, 0, 0, 1])))
            
    return Tinv


def logm_pose(T):
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    omega_hat = logm_rot(R)
    omega = extract_vector_from_skew(omega_hat)
    theta = np.linalg.norm(omega)
    dtwist = np.zeros((4, 4))

    if theta == 0:
        dtwist[0:3, 3] = p
    else:
        G = (np.eye(3) - R) @ omega_hat / theta + np.outer(omega, omega) / theta
        dtwist[0:3, 0:3] = omega_hat
        dtwist[0:3, 3] = np.linalg.solve(G, p) * theta

    return dtwist

def logm_rot(R):
    axis_angle = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    sin_angle = np.linalg.norm(axis_angle)
    cos_angle = (np.trace(R) - 1) / 2

    cos_angle = np.clip(cos_angle, -1, 1)

    if sin_angle < 1e-14:
        alpha = 0.5
    else:
        alpha = np.arctan2(sin_angle, cos_angle) / (2 * sin_angle)

    return (R - R.T) * alpha

def normalize(vector):
    norm_val = np.linalg.norm(vector)
    return vector / norm_val if norm_val != 0 else np.zeros_like(vector)

def pose2quat(T):
    ROT = T[0:3, 0:3, :]
    pos = T[0:3, 3, :]
    quat = rot2quat(ROT)
    return pos, quat.T   # pos is 3xN, quat is 4xN

def quat_conj(q):
    q_new = np.copy(q)
    q_new[1:4] = -q[1:4]
    return q_new

def quat2pose(pos, quat):
    N = pos.shape[1]
    T = np.zeros((4, 4, N))

    for j in range(N):
        # R.from_quat follows the convention of scalar last for quaternions!
        T[0:3, 0:3, j] = R.from_quat(quat[[1, 2, 3, 0],j]).as_matrix()
        T[0:3, 3, j] = pos[:, j]
        T[3, 3, j] = 1

    return T

def rot2quat(R_all):
    N = R_all.shape[2]
    q_all = np.zeros((N, 4))

    for i in range(N):
        R = R_all[:, :, i]
        qs = np.sqrt(np.trace(R) + 1) / 2.0
        kx = R[2, 1] - R[1, 2]
        ky = R[0, 2] - R[2, 0]
        kz = R[1, 0] - R[0, 1]

        if (R[0, 0] >= R[1, 1]) and (R[0, 0] >= R[2, 2]):
            kx1 = R[0, 0] - R[1, 1] - R[2, 2] + 1
            ky1 = R[1, 0] + R[0, 1]
            kz1 = R[2, 0] + R[0, 2]
            add = (kx >= 0)
        elif (R[1, 1] >= R[2, 2]):
            kx1 = R[1, 0] + R[0, 1]
            ky1 = R[1, 1] - R[0, 0] - R[2, 2] + 1
            kz1 = R[2, 1] + R[1, 2]
            add = (ky >= 0)
        else:
            kx1 = R[2, 0] + R[0, 2]
            ky1 = R[2, 1] + R[1, 2]
            kz1 = R[2, 2] - R[0, 0] - R[1, 1] + 1
            add = (kz >= 0)

        if add:
            kx = kx + kx1
            ky = ky + ky1
            kz = kz + kz1
        else:
            kx = kx - kx1
            ky = ky - ky1
            kz = kz - kz1

        nm = np.linalg.norm([kx, ky, kz])
        if nm == 0:
            q = np.array([1, 0, 0, 0])
        else:
            s = np.sqrt(1 - qs**2) / nm
            qv = s * np.array([kx, ky, kz])
            q = np.concatenate(([qs], qv))

            if i > 0 and np.linalg.norm(q - q_all[i - 1, :]) > 0.5:
                q = -q

        q_all[i, :] = q

    return q_all


def calculate_bodytwist_from_poses(T, ds):
    N = T.shape[2]
    screwtwist = np.zeros((6, N))
    
    for k in range(N-1):
        twist_cross = logm_pose(inverse_T(T[:, :, k]) @ T[:, :, k+1]) / ds
        skew_omega = twist_cross[:3, :3]
        screwtwist[:3, k] = extract_vector_from_skew(skew_omega)
        screwtwist[3:6, k] = twist_cross[:3, 3]

    # Copy last sample to maintain the same number of samples as the input T
    screwtwist[:, N-1] = screwtwist[:,N-2]

    return screwtwist



