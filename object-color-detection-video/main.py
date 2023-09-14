import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_WB, 1.0)

# Taking a matrix of size 1 as the kernel
kernel = np.ones((3, 3), np.uint8)


while 1:
    ret, frame = cap.read()
    # Change color format from BGr to HSV
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    into_hsv[:, :, 2] += 42

    # Blue
    L_limit = np.array([98, 50, 50])
    U_limit = np.array([139, 255, 255])
    b_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    b_mask = cv2.erode(b_mask, kernel, iterations=1)
    b_mask = cv2.dilate(b_mask, kernel, iterations=1)

    # Yellow
    L_limit = np.array([15, 50, 150])
    U_limit = np.array([40, 255, 255])
    y_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    y_mask = cv2.erode(y_mask, kernel, iterations=1)
    y_mask = cv2.dilate(y_mask, kernel, iterations=1)

    # Green
    L_limit = np.array([40, 52, 72])
    U_limit = np.array([102, 255, 255])
    g_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    g_mask = cv2.erode(g_mask, kernel, iterations=1)
    g_mask = cv2.dilate(g_mask, kernel, iterations=1)

    # Red
    L_limit = np.array([170, 40, 40])
    U_limit = np.array([180, 255, 255])
    r_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    r_mask = cv2.erode(r_mask, kernel, iterations=2)
    # r_mask = cv2.GaussianBlur(r_mask, (11, 11), 0)
    # _, r_mask = cv2.threshold(r_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    r_mask = cv2.dilate(r_mask, kernel, iterations=2)

    # White
    L_limit = np.array([0, 0, 230])
    U_limit = np.array([255, 255, 255])
    w_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    # w_mask = cv2.erode(w_mask, kernel, iterations=2)
    # w_mask = cv2.GaussianBlur(w_mask, (11, 11), 0)
    # _, w_mask = cv2.threshold(w_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # w_mask = cv2.dilate(w_mask, kernel, iterations=2)

    # Apply original colors to mask.
    blue = cv2.bitwise_and(frame, frame, mask=b_mask)
    green = cv2.bitwise_and(frame, frame, mask=g_mask)
    red = cv2.bitwise_and(frame, frame, mask=r_mask)
    white = cv2.bitwise_and(frame, frame, mask=w_mask)
    cv2.imshow("Original", frame)
    cv2.imshow("Blue Detector", b_mask)
    cv2.imshow("Green Detector", g_mask)
    cv2.imshow("Yellow Detector", y_mask)
    cv2.imshow("Red Detector", r_mask)
    cv2.imshow("White Detector", white)

    if cv2.waitKey(1) == 27:
        break
cap.release()

cv2.destroyAllWindows()
