import cv2
import numpy as np
from pyquaternion import Quaternion ## pip install pyquaternion

# Define camera matrix K
# K: [260.9886474609375, 0.0, 322.07867431640625, 0.0, 260.9886474609375, 179.7025146484375, 0.0, 0.0, 1.0]
fx = 260.9886474609375
fy = 260.9886474609375
cx = 322.07867431640625
cy = 179.7025146484375
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Define 3D points in the world frame

x1, y1, z1 = 2.7129877094525896, -0.2879700118481145, -2.536404399639665
x2, y2, z2 = 1.2915384357003212, -3.4943163535511554, -2.161555826903958
x3, y3, z3 = 2.291253493065628, -4.058678419612267, -1.7340915704909081
x4, y4, z4 = 2.646116147594074, -0.885115759647436, -1.7334489958438823
x5, y5, z5 = 2.014442024597341, -0.9509440747969621, -2.285679631591885
x6, y6, z6 = 2.6338185054577297, -0.23619795461717052, -1.8262379163438438
x7, y7, z7 = 2.1655179782574345, -4.245791735163149, -1.4882541160545077
x8, y8, z8 = 1.7241952278170578, -2.428860112137637, -2.360836270370895
x9, y9, z9 = 2.7106407664894645, -1.5974747231813622, 0.7374993120240866
x10, y10, z10 = 1.4396958941821736, -3.1182177568037712, -1.7416019070094029
x11, y11, z11 = 2.4056629128522387, -2.580867978913784, -1.499505428704834
x12, y12, z12 = 2.395215561031191, -1.966084811818672, -1.5089918965348719


points_3d = np.array([[x1, y1, z1],
                      [x2, y2, z2],
                      [x3, y3, z3],
                        [x4, y4, z4],
                        [x5, y5, z5],
                        [x6, y6, z6],
                        [x7, y7, z7],
                        [x8, y8, z8],
                        [x9, y9, z9],
                        [x10, y10, z10],
                        [x11, y11, z11],
                        [x12, y12, z12]])  # shape should be Nx3


# Given quaternion components
w = 0.9901254394526712
x = -0.039957737686998965
y = -0.08549949403800705
z = -0.1036572711718978

# Quaternion to rotation matrix conversion
q = np.array([w, x, y, z])
q_norm = np.linalg.norm(q)
q = q / q_norm  # Normalize the quaternion

myQuaternion = Quaternion(w, x, y, z)
R = myQuaternion.rotation_matrix
print(f'Rotation matrix..: {R}')
print(R*R.T)

# rotation_matrix = np.array([
#     [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
#     [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
#     [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
# ])

rotation_matrix = np.array([
    [q[0]**2+q[1]**2 - q[2]**2 - q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
    [2*q[1]*q[2] + 2*q[3]*q[0], q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
    [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]
])

print(f'Rotation matrix: {rotation_matrix}')
print(rotation_matrix*rotation_matrix.T)

# Convert rotation matrix to Rodrigues vector
R_vec, _ = cv2.Rodrigues(rotation_matrix)
# rodrigues_vector.flatten()  # Flattening to make it a 1D vector


x = 0.2058384154168687
y = -0.15623392867549235
z = -0.2376815219311228

t_vec = np.array([x, y, z])

# Project points
dist_coeffs = np.zeros(4)  # Assuming no lens distortion
points_2d, _ = cv2.projectPoints(points_3d, R_vec, t_vec, K, dist_coeffs)

print(points_2d)

height = 720
width = 1480

# Initialize a blank image
image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw each point on the image
for point in points_2d:
    x, y = int(point[0][0]), int(point[0][1])
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(image, (x, y), 3, (255, 255, 255), -1)  # Red dot

cv2.imshow("Projected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



