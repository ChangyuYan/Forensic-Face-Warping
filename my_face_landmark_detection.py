import cv2
import dlib

LANDMARK_COUNT = 68

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        for n in range(LANDMARK_COUNT):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break