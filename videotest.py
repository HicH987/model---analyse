import os
from helpers.utils import (
    get_coordinates,
    get_dist,
    pose_detection, 
    get_2d_landmarks,
    make_prediction,
    append_multi,     
    result_window,
    show_prediction,
    main_execution_count,
    )
from helpers.analyse import analyse
from helpers.global_vars import LABELS_2
from helpers.DeepFitClassifier import DeepFitClassifier


import cv2
import mediapipe as mp
from collections import deque
from glob import glob

# --- GLOBAL VAR --------
mp_pose = mp.solutions.pose

clf_model_path = "./models/inf_old_model_4.tflite"
CLF = DeepFitClassifier(clf_model_path, LABELS_2)

K_NUM_FRAMES = 25
frame_queue = deque(maxlen=K_NUM_FRAMES)

output_videos_dir = "outputs_videos"
os.makedirs(output_videos_dir ,exist_ok=1)

count = main_execution_count("helpers/main_count.txt")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = f"{output_videos_dir}/output_video_{count}.avi"
frame_rate = 30.0
frame_size = (640, 480)
video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)


def main():
    threshold = 0.75
    prediction_proba = 0
    predicted_label = ""

    hist_perd = []
    
    
    list_videos = glob("videos_test/bench press/*.mp4")
    # list_videos = glob("videos_test/pullup/*.mp4")
    
    # list_videos = glob("videos_test/arm raise/*.mp4")
    # list_videos = glob("videos_test/biceps curl/*.mp4")
    # list_videos = glob("videos_test/deadlift/*.mp4")
    # list_videos = glob("videos_test/plank/*.mp4")
    # list_videos = glob("videos_test/pushup/*.mp4")
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as POSE:
        for vid_path in list_videos:
            cap = cv2.VideoCapture(vid_path)
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break
                frame = cv2.resize(img,(640, 480), interpolation=cv2.INTER_AREA)
                frame, results = pose_detection(frame, pose_model=POSE)
                landmark_list = get_2d_landmarks(frame, results)
                if len(landmark_list) != 0:
                    predicted_label, prediction_proba = make_prediction(
                        CLF, landmark_list, frame_queue, threshold 
                    )
                        
                    if prediction_proba <= 0.0 :
                        if len(hist_perd) > 0:
                            predicted_label = hist_perd[-1]
                    else:
                        hist_perd.append(predicted_label)
                    
                result_win = result_window(
                    append_multi(
                        [],
                        ("label", predicted_label),
                        ("proba", prediction_proba),
                    )
                )

                cv2.imshow("window", frame)
                cv2.imshow("result_win", result_win)
                
                show_prediction(img, predicted_label, prediction_proba)
                video_writer.write(img)
                
                analyse(results, predicted_label)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
