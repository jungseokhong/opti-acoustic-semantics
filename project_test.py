import numpy as np
import cv2
# # from pyquaternion import Quaternion ## pip install pyquaternion


def parse_data(file_path):
    # Read the entire file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the file content into sections
    sections = content.split('---')
    landmark_points = []
    landmark_classes = []
    position_data = []
    orientation_data = []

    # Process each section
    for section in sections:
        # Extract position and orientation
        if 'pose:' in section:
            position_index = section.find('position:')
            orientation_index = section.find('orientation:')
            landmark_index = section.find('landmark_points:')

            # Extract position
            position_section = section[position_index:orientation_index]
            for line in position_section.split('\n'):
                if 'x:' in line or 'y:' in line or 'z:' in line:
                    value = float(line.split(':')[-1].strip())
                    position_data.append(value)

            # Extract orientation
            orientation_section = section[orientation_index:landmark_index]
            for line in orientation_section.split('\n'):
                if 'x:' in line or 'y:' in line or 'z:' in line or 'w:' in line:
                    value = float(line.split(':')[-1].strip())
                    orientation_data.append(value)

        # Check if the section contains 'landmark_points' and 'landmark_classes'
        if 'landmark_points' in section and 'landmark_classes' in section:
            classes_index = section.find('landmark_classes:')
            points_section = section[section.find('landmark_points:'):classes_index]
            classes_section = section[classes_index:]

            # Extract points
            for line in points_section.split('\n'):
                if 'x:' in line or 'y:' in line or 'z:' in line:
                    value = float(line.split(':')[-1].strip())
                    landmark_points.append(value)

            # Extract classes
            for line in classes_section.split('\n'):
                if '-' in line:
                    landmark_class = line.split('-')[-1].strip()
                    if landmark_class:
                        landmark_classes.append(landmark_class)

    # Convert lists to numpy arrays
    landmark_points_array = np.array(landmark_points).reshape(-1, 3)
    landmark_classes_array = np.array(landmark_classes)
    position_array = np.array(position_data)
    orientation_array = np.array(orientation_data)
    # Reorder the orientation array
    orientation_array = np.roll(orientation_array, -3)

    return position_array, orientation_array, landmark_points_array, landmark_classes_array

# Example usage:
file_path = 'position_data.txt'
position, orientation, landmark_points, landmark_classes = parse_data(file_path)
print("Position:\n", position)
print("Orientation:\n", orientation)
print("Landmark Points:\n", landmark_points)
print("Landmark Classes:\n", landmark_classes)




# Define camera matrix K
# K: [260.9886474609375, 0.0, 322.07867431640625, 0.0, 260.9886474609375, 179.7025146484375, 0.0, 0.0, 1.0]
fx = 260.9886474609375
fy = 260.9886474609375
cx = 322.07867431640625
cy = 179.7025146484375
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Quaternion to rotation matrix conversion
q = orientation
q_norm = np.linalg.norm(q)
q = q / q_norm  # Normalize the quaternion

# myQuaternion = Quaternion(w, x, y, z)
# R = myQuaternion.rotation_matrix
# print(f'Rotation matrix..: {R}')
# print(R*R.T)

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

# camToBody = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
camToBody = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
bodytocam = np.linalg.inv(camToBody)
print(f' Cam to Body: {camToBody} Body to cam: {bodytocam}')
# new_rotation_matrix = np.matmul(rotation_matrix, bodytocam)
new_rotation_matrix = np.matmul(rotation_matrix, camToBody)


# Convert rotation matrix to Rodrigues vector
R_vec, _ = cv2.Rodrigues(new_rotation_matrix)


t_vec = position

# Project points
dist_coeffs = np.zeros(4)  # Assuming no lens distortion
points_2d, _ = cv2.projectPoints(landmark_points, R_vec, t_vec, K, dist_coeffs)

print(points_2d)

height = 360
width = 640

# Initialize a blank image
image = np.zeros((height, width, 3), dtype=np.uint8)

# Draw each point and class label on the image
for i, point in enumerate(points_2d):
    x, y = int(point[0][0]), int(point[0][1])
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green dot
        cv2.putText(image, landmark_classes[i], (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


cv2.imshow("Projected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
