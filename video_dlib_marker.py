import cv2
import dlib

LANDMARK_COUNT = 68

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture("./data/Deepfakes/000_003.mp4")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        print("-------frame: " + str(frame_count) + "------------")
        cv2.imshow('window-name', frame)
        frame_count = frame_count + 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # To speed up detection by converting to grayscale
        faces = detector(frame_gray)
        for face in faces:
            landmarks = predictor(frame_gray, face)
            count = 0
            for n in range(LANDMARK_COUNT):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                print(count, x, y)
                count += 1
    else:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows
