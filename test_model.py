import pickle

import cv2

from utils import get_face_landmarks

emotions = ['HAPPY', 'SAD', 'SURPRISED']

with open('./model', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

check, frame = cap.read()

while True:
    check, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
    if len(face_landmarks) > 0:
        output = model.predict([face_landmarks])
    else:
        continue

    cv2.putText(frame,
                emotions[int(output[0])],
                (10, frame.shape[0] - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                5)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
