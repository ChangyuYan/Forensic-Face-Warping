import cv2
import numpy as np
import dlib

LANDMARK_COUNT = 68
WINDOW_NAME = "Window"

IMG1 = cv2.imread("./sample_images/me.JPG")
IMG1_GRAY = cv2.cvtColor(IMG1, cv2.COLOR_BGR2GRAY)
TEMPLATE = cv2.imread("./sample_images/template.jpg")
TEMPLATE_GRAY = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)

MASK = np.zeros_like(IMG1_GRAY)


def extract_index_np_array(np_array):
    # TODO: This implementation is too ugly; try to optimize this
    idx = None
    for num in np_array[0]:
        idx = num
        break
    return idx


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
faces1 = detector(IMG1_GRAY)


for face in faces1:
    landmarks = predictor(IMG1_GRAY, face)
    landmark_points = []
    for n in range(LANDMARK_COUNT):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))
        # cv2.circle(IMG1, (x, y), 3, (0, 0, 255), -1)

    points = np.array(landmark_points, np.int32)
    convex_hull = cv2.convexHull(points)

    cv2.polylines(IMG1, [convex_hull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(MASK, convex_hull, 255)

    face_image_1 = cv2.bitwise_and(IMG1, IMG1, mask=MASK)

    # Delaunay Triangulation
    rect = cv2.boundingRect(convex_hull)
    sub_div = cv2.Subdiv2D(rect)
    sub_div.insert(landmark_points)
    triangles = sub_div.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indices_triangles = []

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = extract_index_np_array(np.where((points == pt1).all(axis=1)))
        index_pt2 = extract_index_np_array(np.where((points == pt2).all(axis=1)))
        index_pt3 = extract_index_np_array(np.where((points == pt3).all(axis=1)))

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            indices_triangles.append([index_pt1, index_pt2, index_pt3])

        cv2.line(IMG1, pt1, pt2, (0, 0, 255), 2)
        cv2.line(IMG1, pt2, pt3, (0, 0, 255), 2)
        cv2.line(IMG1, pt3, pt1, (0, 0, 255), 2)


# TODO: A lot of repeats here; optimize this
faces2 = detector(TEMPLATE_GRAY)

for face in faces2:
    landmarks = predictor(TEMPLATE_GRAY, face)
    landmark_points = []
    for n in range(LANDMARK_COUNT):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points.append((x, y))

        cv2.circle(TEMPLATE, (x, y), 3, (0, 255, 0), -1)

# Triangulation of the second face, from the first face Delaunay Triangulation

for triangle_index in indices_triangles:
    pt1 = landmark_points[triangle_index[0]]
    pt2 = landmark_points[triangle_index[1]]
    pt3 = landmark_points[triangle_index[2]]

    cv2.line(TEMPLATE, pt1, pt2, (0, 0, 255), 2)
    cv2.line(TEMPLATE, pt2, pt3, (0, 0, 255), 2)
    cv2.line(TEMPLATE, pt3, pt1, (0, 0, 255), 2)


cv2.imshow("Image 1", IMG1)
cv2.imshow("Face Image 1", face_image_1)
cv2.imshow("Template", TEMPLATE)
cv2.imshow("Mask", MASK)
cv2.waitKey(0)
cv2.destroyWindow(WINDOW_NAME)
