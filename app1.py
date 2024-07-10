# Import libraries
import mediapipe as mp
import cv2
import numpy as np
import logging
import mouse
from math import sqrt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from screeninfo import get_monitors

# Screen resolution
screen = get_monitors()[0]
scr_width, scr_height = screen.width, screen.height

# Config draw
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create an HandLandmaker object
base_option = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_option, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options=options)

# Setting camera
cap = cv2.VideoCapture(0)
cap_width = 1920 #1280 # Check your resolution camera
cap_height = 1080 #720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
if not cap.isOpened():
    logging.info("Cannot open camera!")
    exit()

# Width and height of camera
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(w, h)

def cal_distance(a, b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pre_position_mouse = []
while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Normalize image
    image = cv2.resize(image, (scr_width, scr_height), cv2.INTER_LINEAR)
    image = cv2.flip(image, 1)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        # Draw the hand annotations on the image.
        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image,
        #         hand_landmarks,
        #         mp_hands.HAND_CONNECTIONS,
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style())

        # Move mouse
        # print(hand_landmarks.landmark[8])
        hand_landmarks = results.multi_hand_landmarks[0]
        position_mouse = [int(hand_landmarks.landmark[8].x*scr_width),  int(hand_landmarks.landmark[8].y*scr_height)]
        if len(pre_position_mouse) == 0:
            logging.info("Init pre position mouse")
            pre_position_mouse = position_mouse
        else:
            if cal_distance(position_mouse, pre_position_mouse) > 10:
                logging.info(f"Pre position: {pre_position_mouse}")
                logging.info(f"Current position: {position_mouse}")
                mouse.move(position_mouse[0], position_mouse[1], absolute=True, duration=0.01)
                pre_position_mouse = position_mouse

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()