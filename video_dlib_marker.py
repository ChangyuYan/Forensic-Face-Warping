import cv2
import dlib
import os

LANDMARK_COUNT = 68

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

'''
Each video has a corresponding directory,
under which, each file is a single frame,
in which, in line is a landmark coordinate
'''

video_name = "000_003"
output_parent_dir = "output"
os.makedirs(output_parent_dir, exist_ok=True)
output_landmark_dir = os.path.join(output_parent_dir, video_name)
os.makedirs(output_landmark_dir, exist_ok=True)

cap = cv2.VideoCapture("./data/Deepfakes/" + video_name + ".mp4")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    current_frame_landmark = []
    if success:
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
                current_frame_landmark.append([x, y])
                count += 1
        if current_frame_landmark:
            file_name = "frame_" + str(frame_count) + ".txt"
            complete_file_name = os.path.join(output_landmark_dir, file_name)
            file = open(complete_file_name, "w")

            i = 0
            for x, y in current_frame_landmark:
                file.write(str(x) + " " + str(y) + "\n")
                i += 1
            file.close()
    else:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows
