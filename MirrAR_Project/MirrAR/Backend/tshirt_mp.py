import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from tshirt import tshirt
# from necklace import necklace

# For static images:
IMAGE_FILES = []
# num = "1"
# def back(*args):
#   global num
#   tshirt = ["1","2","3","4","5"]
#   num = tshirt.random()
  
with mp_holistic.Holistic(static_image_mode=True) as holistic:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

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
    
    L_hip = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x* width),int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y* height))
    R_hip = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x* width),int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y* height))
    Center = (int(L_Shoulder[0]+R_Shoulder[0]) / 2,int(L_Shoulder[1]+L_hip[1]) / 2)
    
    # Draw landmark annotation on the image.
    # image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x* width),int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y* height))
    # print(R_Shoulder)
    # print(Neck)
    # cv2.createButton("Back",back,None,cv2.QT_PUSH_BUTTON,1)
    
    tshirt(image,Center,L_Shoulder,R_Shoulder,L_hip,R_hip,"1")
    
    # necklace(image,Neck,L_Shoulder,R_Shoulder)
    # print(
    #       f'Nose coordinates: ('
    #       f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x }, '
    #       f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y })'
    #   )
    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()