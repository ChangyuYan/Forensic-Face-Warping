import cv2
import numpy as np
import dlib

LANDMARK_COUNT = 68
WINDOW_NAME = "Window"
IMG1 = cv2.imread("./sample_images/me.JPG")
IMG1_GRAY = cv2.cvtColor(IMG1, cv2.COLOR_BGR2GRAY)
MASK = np.zeros_like(IMG1_GRAY, dtype='uint8')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
faces = detector(IMG1_GRAY)

for face in faces:
    landmarks = predictor(IMG1_GRAY, face)
    landmark_points = []
    for n in range(LANDMARK_COUNT):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
        cv2.circle(IMG1, (x, y), 3, (0, 0, 255), -1)

    landmark_points = np.array(landmark_points, np.int32)
    convex_hull = cv2.convexHull(landmark_points)

    cv2.polylines(IMG1, [convex_hull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(MASK, convex_hull, 255)

    face_image_1 = cv2.bitwise_and(IMG1, IMG1, mask=MASK)


cv2.imshow("Image 1", IMG1)
cv2.imshow("Face Image 1", face_image_1)
cv2.imshow("Mask", MASK)
cv2.waitKey(0)
cv2.destroyWindow(WINDOW_NAME)
