import numpy as np


def euclidean_dist(a, b):
    try:
        if a.shape[1] == 2 and a.shape == b.shape:
            # check if element of a and b is (0,0)
            bol_a = (a[:, 0] != 0).astype(int)
            bol_b = (b[:, 0] != 0).astype(int)
            dist = np.linalg.norm(a - b, axis=1)
            return (dist * bol_a * bol_b).reshape(a.shape[0], 1)
    except:
        print("[Error]: Check dimension of input vector")
        return 0


def norm_X(X):
    num_sample = X.shape[0]
    # Keypoints
    Nose = X[:, 0 * 2 : 0 * 2 + 2]
    Neck = X[:, 1 * 2 : 1 * 2 + 2]
    RHip = X[:, 8 * 2 : 8 * 2 + 2]
    RKnee = X[:, 9 * 2 : 9 * 2 + 2]
    RAnkle = X[:, 10 * 2 : 10 * 2 + 2]
    LHip = X[:, 11 * 2 : 11 * 2 + 2]
    LKnee = X[:, 12 * 2 : 12 * 2 + 2]
    LAnkle = X[:, 13 * 2 : 13 * 2 + 2]
    REye = X[:, 14 * 2 : 14 * 2 + 2]
    LEye = X[:, 15 * 2 : 15 * 2 + 2]
    REar = X[:, 16 * 2 : 16 * 2 + 2]
    LEar = X[:, 17 * 2 : 17 * 2 + 2]

    # Length of head
    length_Neck_LEar = euclidean_dist(Neck, LEar)
    length_Neck_REar = euclidean_dist(Neck, REar)
    length_Neck_LEye = euclidean_dist(Neck, LEye)
    length_Neck_REye = euclidean_dist(Neck, REye)
    length_Nose_LEar = euclidean_dist(Nose, LEar)
    length_Nose_REar = euclidean_dist(Nose, REar)
    length_Nose_LEye = euclidean_dist(Nose, LEye)
    length_Nose_REye = euclidean_dist(Nose, REye)
    length_head = np.maximum.reduce(
        [
            length_Neck_LEar,
            length_Neck_REar,
            length_Neck_LEye,
            length_Neck_REye,
            length_Nose_LEar,
            length_Nose_REar,
            length_Nose_LEye,
            length_Nose_REye,
        ]
    )

    # Length of torso
    length_Neck_LHip = euclidean_dist(Neck, LHip)
    length_Neck_RHip = euclidean_dist(Neck, RHip)
    length_torso = np.maximum(length_Neck_LHip, length_Neck_RHip)

    # Length of right leg
    length_leg_right = euclidean_dist(RHip, RKnee) + euclidean_dist(RKnee, RAnkle)

    # Length of left leg
    length_leg_left = euclidean_dist(LHip, LKnee) + euclidean_dist(LKnee, LAnkle)

    # Length of leg
    length_leg = np.maximum(length_leg_right, length_leg_left)

    # Length of body
    length_body = length_head + length_torso + length_leg

    # Check all samples have length_body of 0
    length_chk = (length_body > 0).astype(int)

    # Check keypoints at origin
    keypoints_chk = (X > 0).astype(int)

    chk = length_chk * keypoints_chk

    # Set all length_body of 0 to 1 (to avoid division by 0)
    length_body[length_body == 0] = 1

    # The center of gravity
    # number of point OpenPose locates:
    num_pts = (X[:, 0::2] > 0).sum(1).reshape(num_sample, 1)
    centr_x = X[:, 0::2].sum(1).reshape(num_sample, 1) / num_pts
    centr_y = X[:, 1::2].sum(1).reshape(num_sample, 1) / num_pts

    # The  coordinates  are  normalized relative to the length of the body and the center of gravity
    X_norm_x = (X[:, 0::2] - centr_x) / length_body
    X_norm_y = (X[:, 1::2] - centr_y) / length_body

    # Stack 1st element x and y together
    X_norm = np.column_stack((X_norm_x[:, :1], X_norm_y[:, :1]))

    for i in range(1, X.shape[1] // 2):
        X_norm = np.column_stack(
            (X_norm, X_norm_x[:, i : i + 1], X_norm_y[:, i : i + 1])
        )

    # Set all samples have length_body of 0 to origin (0, 0)
    X_norm = X_norm * chk

    return X_norm
