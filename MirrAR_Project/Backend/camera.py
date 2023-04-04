import cv2
import numpy as np
import imutils
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from necklace import necklace

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FPS, 30)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            while self.video.isOpened():
                success, image = self.video.read()
                
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = holistic.process(image)
                L_Shoulder = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x* width),int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y* height))
                R_Shoulder = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x* width),int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y* height))
                Neck =  (int(L_Shoulder[0]+R_Shoulder[0]) / 2,int(L_Shoulder[1]+R_Shoulder[1]) / 2)
                
                # Draw landmark annotation on the image.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                necklace(image,Neck,L_Shoulder,R_Shoulder)
                _, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()
