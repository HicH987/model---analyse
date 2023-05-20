LABELS = [
    "arm raise",
    "basic curl",
    "biceps curl bar",
    "bicyclecrunch",
    "birddog",
    "deadlift",
    "fly",
    "leg raise",
    "overhead press",
    "plank",
    "pushup",
    "russian twist",
    "squat",
]


# used landmarks
LANDMARKS_DICT = {
    0: 0,  # nose
    1: 10,  # mouth_right # remplace "neck" point of the coco pose model
    2: 12,  # right_shoulder
    3: 14,  # right elbow
    4: 16,  #  right wrist
    5: 11,  #  left_shoulder
    6: 13,  #  left_elbow
    7: 15,  #  left_wrist
    8: 24,  #  right_hip
    9: 26,  #  right_knee
    10: 28,  #  right_ankle
    11: 23,  #  left_hip
    12: 25,  #  left_knee
    13: 27,  #  left_ankle
    14: 5,  #  right eye
    15: 2,  #  left_eye
    16: 8,  #  right ear
    17: 7,  #  left_ear
}
