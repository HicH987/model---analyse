import cv2
import numpy as np
import time

import mediapipe as mp

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    # check cord sys area
    if angle > 180.0:
        angle = 360 - angle

    return angle


def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility,
    ]


counter = 0  # movement of exercise
status = True
l_status = True
r_status = True
action = "no"  # state of move

past_exercice = None

nochrono = True
t = "00:00"

start_time_total = time.time()
total_chrono = None
t_chrono = None
start_time = None


def analyse(mdp_results, nom_exercice):
    global counter,l_status,r_status,status, action, past_exercice, nochrono,t, start_time_total, total_chrono, t_chrono, start_time
    
    total_chrono = time.time() - start_time_total
    minutes_t = int((total_chrono % 3600) / 60)
    seconds_t = int(total_chrono % 60)
    t_chrono = f"{minutes_t:02d}:{seconds_t:02d}"

    try:
        if mdp_results.pose_landmarks:
            landmarks = mdp_results.pose_landmarks.landmark
            if nom_exercice != past_exercice:
                counter = 0  # movement of exercise
                status = True
                l_status = True
                r_status = True
                action = "no"
                nochrono =True
                
            if nom_exercice == "pushup":
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")
                r_angle_bras = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle_bras = calculate_angle(l_shoulder, l_elbow, l_wrist)

                avg_arm_angle = (l_angle_bras + r_angle_bras) // 2

                if status:
                    if avg_arm_angle < 100:
                        counter = counter + 1
                        action = "UP"

                        status = False
                else:
                    if avg_arm_angle > 150:
                        action = "DOWN"
                        status = True
                past_exercice = "pushup"

            if nom_exercice == "pullup":
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")
                l_hip = detection_body_part(landmarks, "LEFT_WRIST")
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")

                r_angle_bras = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle_bras = calculate_angle(l_shoulder, l_elbow, l_wrist)

                avg_arm_angle = (l_angle_bras + r_angle_bras) // 2

                if status:
                    if avg_arm_angle < 25:
                        counter = counter + 1
                        action = "DOWN"

                        status = False
                else:
                    if avg_arm_angle > 155:
                        action = "UP"
                        status = True
                past_exercice = "pullup"

            if nom_exercice == "plank":
                if nochrono:
                    start_time = time.time()
                    nochrono = False
                    # print(nochrono)
                if not nochrono:
                    try:
                        elapsed_time = time.time() - start_time

                        minutes = int((elapsed_time % 3600) / 60)
                        seconds = int(elapsed_time % 60)

                        t = f"{minutes:02d}:{seconds:02d}"
                        # print(t)
                    except:
                        print("Error")    
                past_exercice = "plank"

            if nom_exercice == "squat":
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_knee = detection_body_part(landmarks, "LEFT_KNEE")
                l_ankle = detection_body_part(landmarks, "LEFT_ANKLE")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
                r_ankle = detection_body_part(landmarks, "RIGHT_ANKLE")

                angle4 = calculate_angle(r_hip, r_knee, r_ankle)
                # # print(angle4)
                angle5 = calculate_angle(l_hip, l_knee, l_ankle)
                # # print(angle5+"L")

                angle = (angle4 + angle5) // 2
                # print(angle)

                if status:
                    if angle < 110:
                        counter = counter + 1
                        action = "UP"
                        status = False
                else:
                    if angle > 165:
                        action = "DOWN"
                        status = True
                past_exercice = "squat"

            if nom_exercice == "overhead press":
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                angle4 = calculate_angle(r_hip, r_shoulder, r_elbow)
                # # print(angle4)
                angle5 = calculate_angle(l_hip, l_shoulder, l_elbow)
                # # print(angle5+"L")

                angle = (angle4 + angle5) // 2
                # print(angle)

                if status:
                    if angle > 150:
                        action = "DOWN"
                        status = False

                else:
                    if angle < 95:
                        counter = counter + 1
                        action = "UP"
                        status = True
                past_exercice = "overhead press"

            if nom_exercice == "deadlift":
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_knee = detection_body_part(landmarks, "LEFT_KNEE")
                l_ankle = detection_body_part(landmarks, "LEFT_ANKLE")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
                r_ankle = detection_body_part(landmarks, "RIGHT_ANKLE")

                angle4 = calculate_angle(r_hip, r_knee, r_ankle)
                # # print(angle4)
                angle5 = calculate_angle(l_hip, l_knee, l_ankle)
                # # print(angle5+"L")

                angle = (angle4 + angle5) // 2
                # print(angle)

                if status:
                    if angle < 110:
                        counter = counter + 1
                        action = "UP"
                        status = False
                else:
                    if angle > 165:
                        action = "DOWN"
                        status = True
                past_exercice = "deadlift"

            if nom_exercice == "bench press":
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")

                r_angle_bras = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle_bras = calculate_angle(l_shoulder, l_elbow, l_wrist)

                avg_arm_angle = (l_angle_bras + r_angle_bras) // 2
                # print(avg_arm_angle)

                if status:
                    if avg_arm_angle < 72:
                        counter = counter + 1
                        action = "UP"
                        status = False
                else:
                    if avg_arm_angle > 155:
                        action = "DOWN"
                        status = True

                past_exercice = "bench press"

            if nom_exercice == "biceps curl bar":
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")

                r_angle_bras = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle_bras = calculate_angle(l_shoulder, l_elbow, l_wrist)

                avg_arm_angle = (l_angle_bras + r_angle_bras) // 2

                # print(avg_arm_angle)

                if status:
                    if avg_arm_angle > 163:
                        action = "UP"
                        status = False

                else:
                    if avg_arm_angle < 30:
                        counter = counter + 1
                        action = "DOWN"
                        status = True

                past_exercice = "biceps curl bar"

            if nom_exercice == "arm raise":
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                angle4 = calculate_angle(r_hip, r_shoulder, r_elbow)
                # # print(angle4)
                angle5 = calculate_angle(l_hip, l_shoulder, l_elbow)
                # # print(angle5+"L")

                angle = (angle4 + angle5) // 2
                # print(angle)

                if status:
                    if angle < 16:
                        action = "UP"
                        status = False
                else:
                    if angle > 99:
                        counter = counter + 1
                        action = "DOWN"
                        status = True

                past_exercice = "arm raise"

            if nom_exercice == "leg raise":
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_knee = detection_body_part(landmarks, "LEFT_KNEE")
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")

                angle4 = calculate_angle(r_knee, r_hip, r_shoulder)
                # # print(angle4)
                angle5 = calculate_angle(l_knee, l_hip, l_shoulder)
                # # print(angle5+"L")

                angle = (angle4 + angle5) // 2
                # print(angle)

                if status:
                    if angle < 72:
                        action = "DOWN"

                        counter = counter + 1
                        status = False
                else:
                    if angle > 158:
                        action = "UP"

                        status = True
                past_exercice = "leg raise"

            if nom_exercice == "birddog":
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_knee = detection_body_part(landmarks, "LEFT_KNEE")
                l_ankle = detection_body_part(landmarks, "LEFT_ANKLE")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
                r_ankle = detection_body_part(landmarks, "RIGHT_ANKLE")

                angle4 = calculate_angle(r_hip, r_knee, r_ankle)
                # # print(angle4)
                angle5 = calculate_angle(l_hip, l_knee, l_ankle)
                # # print(angle5+"L")

                angle = (angle4 + angle5) // 2
                # print(angle)

                if status:
                    if angle < 51:
                        action = "UP"
                        status = False
                else:
                    if angle > 110:
                        counter = counter + 1
                        action = "DOWN"
                        status = True

                past_exercice = "birddog"

            if nom_exercice == "basic curl":
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")

                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                
                r_angle_bras = calculate_angle(r_shoulder, r_elbow, r_wrist)
                l_angle_bras = calculate_angle(l_shoulder, l_elbow, l_wrist)

                avg_arm_angle = (l_angle_bras + r_angle_bras) // 2
                
                if l_status and l_angle_bras > 160:
                    counter += 1
                    action = "UP Left"
                    l_status = False
                elif not l_status and l_angle_bras < 50:
                    action = "NEXT HAND"
                    l_status = True 
                    
                if r_status and r_angle_bras > 160:
                    counter += 1
                    action = "UP Right"
                    r_status = False
                elif not r_status and r_angle_bras < 50:
                    action = "NEXT HAND"
                    r_status = True 

                # if status:
                #     if avg_arm_angle > 165:
                #         action = "UP"
                #         status = False

                # else:
                #     if avg_arm_angle < 105:
                #         counter = counter + 1
                #         action = "NEXT HAND"
                #         status = True

                past_exercice = "basic curl"
                
            if nom_exercice == "bicyclecrunch":
                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_knee = detection_body_part(landmarks, "LEFT_KNEE")
                l_ankle = detection_body_part(landmarks, "LEFT_ANKLE")
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_knee = detection_body_part(landmarks, "RIGHT_KNEE")
                r_ankle = detection_body_part(landmarks, "RIGHT_ANKLE")

                # angle4 = calculate_angle(r_hip, r_knee, r_ankle)
                # angle5 = calculate_angle(l_hip, l_knee, l_ankle)

                # angle = (angle4 + angle5) // 2


                # if status:
                #     if angle < 70:
                #         action = "PUSH"
                #         status = False
                # else:
                #     if angle > 96:
                #         counter = counter + 1
                #         action = "NEXT LEG"
                #         status = True
                r_angle_bras = calculate_angle(r_hip, r_knee, r_ankle)
                l_angle_bras = calculate_angle(l_hip, l_knee, l_ankle)
                
                if l_status and l_angle_bras < 70:
                    counter += 1
                    action = "Push Right"
                    l_status = False
                elif not l_status and l_angle_bras  > 96:
                    action = "NEXT LEG"
                    l_status = True 
                    
                if r_status and r_angle_bras < 70:
                    counter += 1
                    action = "Push Left"
                    r_status = False
                elif not r_status and r_angle_bras > 96:
                    action = "NEXT LEG"
                    r_status = True 
                past_exercice = "bicyclecrunch"
                
            if nom_exercice == "fly":
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_shoulder = detection_body_part(landmarks, "RIGHT_SHOULDER")
                r_elbow = detection_body_part(landmarks, "RIGHT_ELBOW")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")

                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_shoulder = detection_body_part(landmarks, "LEFT_SHOULDER")
                l_elbow = detection_body_part(landmarks, "LEFT_ELBOW")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                # r_x_dist = abs(r_shoulder[0] - r_wrist[0])
                # l_x_dist = abs(l_shoulder[0] - l_wrist[0])
                
                r_angle_bras = calculate_angle(r_hip, r_shoulder, r_elbow)
                l_angle_bras = calculate_angle(l_hip, l_shoulder, l_elbow)


                # if status and (r_x_dist < 0.09 or l_x_dist< 0.09 ):
                if status and (r_angle_bras > 20 or l_angle_bras > 20 ):
                        action = "Open"
                        status = False
                        # print(r_x_dist, l_x_dist)
                        # print(r_angle_bras, l_angle_bras)
                # elif not status and (r_x_dist > 0.15 or l_x_dist > 0.15 ):
                elif not status and (r_angle_bras < 20 or l_angle_bras < 20 ):
                        counter = counter + 1
                        action = "Close"
                        status = True
                        # print(r_x_dist, l_x_dist)
                        # print(r_angle_bras, l_angle_bras)

                past_exercice = "fly"
                
            if nom_exercice == "russian twist":
                r_hip = detection_body_part(landmarks, "RIGHT_HIP")
                r_wrist = detection_body_part(landmarks, "RIGHT_WRIST")

                l_hip = detection_body_part(landmarks, "LEFT_HIP")
                l_wrist = detection_body_part(landmarks, "LEFT_WRIST")

                r_x_dist = abs(r_hip[0] - l_wrist[0])
                l_x_dist = abs(l_hip[0] - r_wrist[0])
                


                if l_status and (r_x_dist < 0.09 ):
                        counter = counter + 1
                        action = "Next Hip"
                        l_status = False

                elif not l_status and (r_x_dist > 0.15 ):
                        action = "Next Hip"
                        l_status = True
                if r_status and (l_x_dist < 0.09 ):
                        counter = counter + 1
                        action = "Next Hip"
                        r_status = False

                elif not r_status and (l_x_dist > 0.15 ):
                        action = "Next Hip"
                        r_status = True


                past_exercice = "russian twist"

            display_analyse_result(nom_exercice)
    except:
        print("FREEZE")

def display_analyse_result(
    nom_exercice,
):
    global nochrono, counter, action, t_chrono, t
    
    score_table = np.ones((210,360), dtype=np.uint8)
    cv2.putText(
        score_table,
        "Activity : " + nom_exercice,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (182, 158, 128),
        2,
        cv2.LINE_AA,
    )

    if nochrono:
        cv2.putText(
            score_table,
            "Counter : " + str(counter),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (182, 158, 128),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            score_table,
            "Action : " + str(action),
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (182, 158, 128),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            score_table,
            "Chrono Total : " + str(t_chrono),
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (182, 158, 128),
            2,
            cv2.LINE_AA,
        )

    else:
        cv2.putText(
            score_table,
            "Chrono : " + str(t),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (182, 158, 128),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Score Table", score_table)
