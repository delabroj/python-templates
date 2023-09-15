import cv2
import numpy as np

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_AUTO_WB, 0.0)  # Disable automatic white balance
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask.copy(), cv2.MORPH_OPEN, kernel, iterations=2)

    fgmask = cv2.dilate(fgmask, kernel, iterations=5)

    # Apply median blur
    fgmask = cv2.medianBlur(fgmask, 1)

    # Filter frame
    filtered = cv2.bitwise_and(frame, frame, mask=fgmask)

    cv2.imshow("Camera", frame)
    cv2.imshow("Motion Detection", fgmask)
    cv2.imshow("Filtered", filtered)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
