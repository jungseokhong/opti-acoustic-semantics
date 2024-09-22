import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

def spherical_to_cartesian(elev, azim):
    """ Convert spherical coordinates (elevation and azimuth) to Cartesian coordinates. """
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    x = np.cos(elev_rad) * np.cos(azim_rad)
    y = np.cos(elev_rad) * np.sin(azim_rad)
    z = np.sin(elev_rad)
    return np.array([x, y, z])

def calculate_offset(position, view_direction, offset_magnitude):
    """ Calculate and return an offset for a given position in the view direction. """
    distance = np.dot(position, view_direction)
    return view_direction * offset_magnitude * (1 + distance)  # Adjust offset based on distance

def extract_unique_classes(json_file_path):
    """ Extract unique classes from the entire JSON file and return a dictionary with class-to-number mapping. """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    all_landmark_classes = []
    for frame_data in data:
        all_landmark_classes.extend(frame_data['landmark_classes'])

    unique_classes = sorted(list(set(all_landmark_classes)))
    class_to_number = {landmark_class: i + 1 for i, landmark_class in enumerate(unique_classes)}

    return class_to_number, unique_classes

def plot_3d_landmarks_and_robot_poses3(json_file_path, frame_num, output_folder):
    # Extract unique classes and create a class-to-number mapping
    class_to_number, unique_classes = extract_unique_classes(json_file_path)

    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get unique colors for each landmark class using the 'tab20' colormap
    colors = plt.cm.get_cmap('hsv', len(unique_classes))
    class_to_color = {landmark_class: colors(i) for i, landmark_class in enumerate(unique_classes)}


    frames_to_draw_dict = {}




    for frame in range(0, frame_num + 1):  # Loop through frames up to frame_num

        # Loop through the data and add only unique frame_num entries to the dictionary
        for frame_data in data:
            frame_num = frame_data['frame_num']
            if frame_num not in frames_to_draw_dict:
                frames_to_draw_dict[frame_num] = frame_data

        # Convert the dictionary values back to a list to use it in your logic
        frames_to_draw = [frame_data for frame_num, frame_data in frames_to_draw_dict.items() if frame_num <= frame]

        # frames_to_draw = [frame_data for frame_data in data if frame_data['frame_num'] <= frame]
        # print(f'frame: {frame} frames_to_draw: {frames_to_draw}')
        if not frames_to_draw:
            print(f"No data found for frame range 0 to {frame}.")
            continue

        # Extract the landmark points and classes for the current frame
        current_frame_data = next((frame_data for frame_data in data if frame_data['frame_num'] == frame), None)
        # print(f'frame: {frame} current_frame_data: {current_frame_data}')
        if not current_frame_data:
            print(f"No data found for frame {frame}.")
            continue

        landmark_points = current_frame_data['landmark_points']
        landmark_classes = current_frame_data['landmark_classes']

        X = [point[0] for point in landmark_points]
        Y = [point[1] for point in landmark_points]
        Z = [point[2] for point in landmark_points]

        fig = plt.figure()
        fig.set_size_inches(12, 8)
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        ax.set_axis_off()

        elevation = 13
        azimuth = 174
        view_direction = spherical_to_cartesian(elevation, azimuth)

        initial_offset_magnitude = 0.5
        min_distance = 0.3
        label_positions = [np.array([X[i], Y[i], Z[i]]) for i in range(len(X))]

        for i in range(len(X)):
            landmark_class = landmark_classes[i]
            class_color = class_to_color[landmark_class]
            landmark_number = class_to_number[landmark_class]

            initial_offset = calculate_offset([X[i], Y[i], Z[i]], view_direction, initial_offset_magnitude)
            new_position = np.array([X[i], Y[i], Z[i]]) + initial_offset

            offset_magnitude = initial_offset_magnitude
            while any(np.linalg.norm(new_position - pos) < min_distance for j, pos in enumerate(label_positions) if i != j):
                offset_magnitude += 1
                additional_offset = calculate_offset([X[i], Y[i], Z[i]], view_direction, offset_magnitude)
                new_position = np.array([X[i], Y[i], Z[i]]) - additional_offset

            label_positions[i] = new_position
            ax.scatter(X[i], Y[i], Z[i], color=class_color, s=20)
            ax.text(new_position[0], new_position[1], new_position[2], str(landmark_number), color='black', fontsize=12)

        for frame_data in frames_to_draw:
            position = np.array(frame_data['position'])
            rotation_matrix = np.array(frame_data['rotation_matrix'])
            ax_length = 0.05 if frame_data['frame_num'] != frame else 0.3
            x_axis = np.array([ax_length, 0, 0])
            y_axis = np.array([0, ax_length, 0])
            z_axis = np.array([0, 0, ax_length])

            x_axis_rotated = np.dot(rotation_matrix, x_axis) + position
            y_axis_rotated = np.dot(rotation_matrix, y_axis) + position
            z_axis_rotated = np.dot(rotation_matrix, z_axis) + position

            alpha_val = 0.3 if frame_data['frame_num'] != frame else 1
            ax.quiver(position[0], position[1], position[2], x_axis_rotated[0] - position[0], x_axis_rotated[1] - position[1], x_axis_rotated[2] - position[2], color='r', alpha=alpha_val)
            ax.quiver(position[0], position[1], position[2], y_axis_rotated[0] - position[0], y_axis_rotated[1] - position[1], y_axis_rotated[2] - position[2], color='g', alpha=alpha_val)
            ax.quiver(position[0], position[1], position[2], z_axis_rotated[0] - position[0], z_axis_rotated[1] - position[1], z_axis_rotated[2] - position[2], color='b', alpha=alpha_val)

        # Number of landmarks in the current frame
        num_landmarks = len(landmark_points)

        # Update the title with number of landmarks
        # if frame == 316:
        #     ax.set_title(f'Number of Landmarks: {num_landmarks}')

        ax.set_title(f'3D Landmark Plot and Robot Poses for Frame {frame} | Number of Landmarks: {num_landmarks}')
        ax.view_init(elev=13, azim=174)
        ax.set_box_aspect([0.3, 2, 1])

        # Determine which classes are present in the current frame
        present_classes = set(landmark_classes)
        
        # Create legend with grayed-out colors for classes not present in the current frame
        legend_items = [
            mpatches.Patch(
                color=class_to_color[cls] if cls in present_classes else 'lightgray', 
                label=f"{class_to_number[cls]}: {cls}"
            ) 
            for cls in unique_classes
        ]




        legend = ax.legend(handles=legend_items, title="Number: Landmark Class", loc='center', bbox_to_anchor=(0.5, -0.15), ncol=4)

        # Check if the legend exists and modify the text color for absent classes
        # if legend:
        #     for text in legend.get_texts():
        #         cls_name = text.get_text().split(": ")[1]
        #         if cls_name not in present_classes:
        #             text.set_color("white")  # Change font color to white for absent classes

        plt.subplots_adjust(bottom=0.3)

        # Save the current frame as a PNG image
        plt.savefig(f'{output_folder}/frame_{frame:04d}.png')
        plt.close(fig)

    print(f"Frames saved in {output_folder}")



# Example usage
# json_file_path = '/home/jungseok/Downloads/landmarks_try4.json'
json_file_path = '/home/jungseok/Downloads/llm_data/llm_filter_output_20obj2_0.57_0.1_spacy/labels/landmarks.json'
frame_num = 500  # Specify the frame number
output_folder = '/home/jungseok/Downloads/landmark_frames'

plot_3d_landmarks_and_robot_poses3(json_file_path, frame_num, output_folder)
