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

    # cv2.polylines(IMG1, [convex_hull], True, (255, 0, 0), 3)
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


# TODO: A lot of repeats here; optimize this
faces2 = detector(TEMPLATE_GRAY)


for face in faces2:
    landmarks = predictor(TEMPLATE_GRAY, face)
    landmark_points2 = []
    for n in range(LANDMARK_COUNT):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_points2.append((x, y))

        # cv2.circle(TEMPLATE, (x, y), 3, (0, 255, 0), -1)

# Triangulation of both faces
for triangle_index in indices_triangles:
    # Triangulation of the first face
    tr1_pt1 = landmark_points[triangle_index[0]]
    tr1_pt2 = landmark_points[triangle_index[1]]
    tr1_pt3 = landmark_points[triangle_index[2]]

    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    # cv2.rectangle(IMG1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped_triangle = IMG1[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)

    points1 = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]])

    cv2.fillConvexPoly(cropped_tr1_mask, points1, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

    cv2.line(IMG1, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
    cv2.line(IMG1, tr1_pt2, tr1_pt3, (0, 0, 255), 2)
    cv2.line(IMG1, tr1_pt3, tr1_pt1, (0, 0, 255), 2)

    # Triangulation of the second face, from the first face Delaunay Triangulation
    tr2_pt1 = landmark_points2[triangle_index[0]]
    tr2_pt2 = landmark_points2[triangle_index[1]]
    tr2_pt3 = landmark_points2[triangle_index[2]]

    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2

    cropped_triangle2 = TEMPLATE[y: y + h, x: x + w]

    cropped_tr2_mask = np.zeros((h, w), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]])

    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

    cv2.line(TEMPLATE, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
    cv2.line(TEMPLATE, tr2_pt2, tr2_pt3, (0, 0, 255), 2)
    cv2.line(TEMPLATE, tr2_pt3, tr2_pt1, (0, 0, 255), 2)

    # Warp Triangles

    points1 = np.float32(points1)
    points2 = np.float32(points2)

    M = cv2.getAffineTransform(points1, points2)
    warp_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))

    break

cv2.imshow("Image 1", IMG1)
cv2.imshow("Template", TEMPLATE)
cv2.imshow("Cropped Triangle 1", cropped_triangle)
cv2.imshow("Cropped Triangle 2", cropped_triangle2)
cv2.imshow("Warped Triangle", warp_triangle)
cv2.waitKey(0)
cv2.destroyWindow(WINDOW_NAME)
