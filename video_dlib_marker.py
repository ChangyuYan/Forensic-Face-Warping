import cv2

cap = cv2.VideoCapture("./data/Deepfakes/000_003.mp4")

count = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:
        cv2.imshow('window-name', frame)
        # cv2.imwrite("frame%d.jpg" % count, frame)
        count = count + 1
    else:
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows
