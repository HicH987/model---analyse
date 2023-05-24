import math
import os
import cv2
import mediapipe as mp
from helpers.global_vars import LANDMARKS_DICT
from helpers.normlization import norm_X

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def init_pose(model_complexity):
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def pose_detection(image, pose_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = pose_model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(img, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )

    return img


def get_2d_landmarks(img, results):
    landmark_list = []
    if results.pose_landmarks:
        height, width, _ = img.shape
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_pixel_x, landmark_pixel_y = (
                int(landmark.x * width),
                int(landmark.y * height),
            )
            landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])

    return landmark_list


import numpy as np


def get_row_landmarks(results):
    landmark_list = []
    if results.pose_landmarks:
        landmark_list = np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
    return landmark_list


def normalize_row(landmark_list):
    inp_pushup = _get_targetd_landmarks(landmark_list)
    X_sample = np.array(inp_pushup).reshape(1, -1)
    X_sample_norm = norm_X(X_sample)
    return X_sample_norm


def make_prediction(clf, landmark_list, frame_queue, threshold=0.5):
    inp_pushup = _get_targetd_landmarks(landmark_list)
    predicted_label, prediction_proba = clf.predict(inp_pushup)
    if prediction_proba >= threshold:
        frame_queue.append(predicted_label)
        predicted_label_smoothed = max(set(frame_queue), key=frame_queue.count)
        return predicted_label_smoothed, prediction_proba
    return "", 0.0
    # frame_queue.append(predicted_label)
    # predicted_label_smoothed = max(set(frame_queue), key=frame_queue.count)
    # return predicted_label_smoothed, prediction_proba


def _get_targetd_landmarks(landmark_list):
    inp_pushup = []
    for index in range(0, 36):
        if index < 18:
            if index == 1:
                nose = landmark_list[0][1:]
                l_shoulder = landmark_list[11][1:]
                r_shoulder = landmark_list[12][1:]
                neck = _get_neck_point(nose, l_shoulder, r_shoulder)
                inp_pushup.append(round(neck[0], 3))
            else:
                inp_pushup.append(round(landmark_list[LANDMARKS_DICT[index]][1], 3))
        else:
            if index - 18 == 1:
                inp_pushup.append(round(neck[1], 3))
            else:
                inp_pushup.append(
                    round(landmark_list[LANDMARKS_DICT[index - 18]][2], 3)
                )
    return inp_pushup


def _get_neck_point(nose, l_shoulder, r_shoulder):
    # Retrieve the x and y coordinates of the left and right shoulders and the midpoint between them
    left_shoulder_x = l_shoulder[0]
    left_shoulder_y = l_shoulder[1]
    right_shoulder_x = r_shoulder[0]
    right_shoulder_y = r_shoulder[1]
    # Retrieve the y coordinate of the nose
    nose_y = nose[1]

    shoulder_midpoint_x = (left_shoulder_x + right_shoulder_x) / 2
    shoulder_midpoint_y = (left_shoulder_y + right_shoulder_y) / 2

    # Calculate the average y coordinates of the shoulders and the nose/midpoint
    shoulderMidpointY_noseY_midpoint_y = (nose_y + shoulder_midpoint_y) / 2

    # Calculate the neck position
    neck_position = (shoulder_midpoint_x, shoulderMidpointY_noseY_midpoint_y)

    return neck_position


def show_prediction(img, pred_label, pred_proba):
    cv2.rectangle(img, (0, 0), (300, 40), (245, 117, 16), -1)

    cv2.putText(
        img,
        f"prediction_class: {pred_label}",
        (10, 15),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"prediction_prob: { pred_proba  }",
        (10, 35),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def resize_image(img, scale_percent):
    # Calculate the new dimensions of the image
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Return the resized image
    return resized_img


def result_window(messages):
    num_msg = len(messages)
    y_msg = 50
    black_img = np.zeros((num_msg*2*y_msg, 500), dtype=np.uint8)
    for lbl, msg in messages:
        
        cv2.putText(
            black_img,
            f"{lbl}: {msg}",
            (10, y_msg),
            cv2.FONT_HERSHEY_COMPLEX,
            1.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_msg+=50

    return black_img

def append_multi(lst, *elements):
    """
    Append multiple elements to a list at once.

    Args:
        lst (list): List to append elements to.
        *elements: Variable number of elements to append to the list.

    Returns:
        list: The updated list after appending all the elements.
    """
    lst.extend(elements)
    return lst

def main_execution_count(count_file_path):
    # Define the path to the file that stores the count
    # count_file_path = 'models/models_count.txt'

    # Check if the count file exists
    if os.path.exists(count_file_path):
        # If it does, read the current count from the file
        with open(count_file_path, 'r') as f:
            count = int(f.read().strip())
    else:
        # If it doesn't, start the count at 0
        count = 0

    # Increment the count
    count += 1
    
    # Write the updated count to the file
    with open(count_file_path, 'w') as f:
        f.write(str(count))
        
    return count




def get_dist(img, landmark_list, point1, point2, draw=True):
    height, width, _ = img.shape

    x1, y1 = landmark_list[point1][1:]
    x2, y2 = landmark_list[point2][1:]

    dist = math.sqrt(((x2 - x1) / width) ** 2 + ((y2 - y1) / height) ** 2)

    if draw:
        # Drawing lines between the three points
        cv2.line(img, (x1, y1), (x2, y2), (255, 5, 255), 3)

        # Drawing circles at intersection points of lines
        cv2.circle(img, (x1, y1), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (75, 0, 130), 2)
        cv2.circle(img, (x2, y2), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (75, 0, 130), 2)

        # Show angles between lines
        i = cv2.putText(
            img,
            str(round(dist, 2)),
            (x2 - 50, y2 + 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
        )
    return dist, i



def get_coordinates(img, landmark_list, point, draw=True):
    height, width, _ = img.shape

    x, y = landmark_list[point][1:]

    if draw:
        # Drawing circles at intersection points of lines
        cv2.circle(img, (x, y), 5, (75, 0, 130), cv2.FILLED)
        cv2.circle(img, (x, y), 15, (75, 0, 130), 2)


        # Show angles between lines
        i = cv2.putText(
            img,
            f'{x} / {y}',
            (x - 50, y + 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            2,
        )
    return (x,y), i
