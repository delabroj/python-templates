# Import required libraries
import cv2
import numpy as np
import dlib
import inspect


# Connects to your computer's default camera
cap = cv2.VideoCapture(0)


# Create the correlation tracker - the object needs to be initialized
# before it can be used
tracker = dlib.correlation_tracker()


# Capture frames continuously
k = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if k == 0:
        # Start a track on the juice box. If you look at the first frame you
        # will see that the juice box is contained within the bounding
        # box (74, 67, 112, 153).
        tracker.start_track(gray, dlib.rectangle(74, 67, 112, 153))
        k = 1
    else:
        # Else we just attempt to track from the previous frame
        tracker.update(gray)

    # Get the coordinates of faces
    pos = tracker.get_position()
    x = int(pos.tl_corner().x)
    y = int(pos.tl_corner().y)
    x1 = int(pos.br_corner().x)
    y1 = int(pos.br_corner().y)

    print(x, x1, y, y1)
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

    # Display the box and faces
    cv2.putText(
        frame,
        "thing",
        (10, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    # print(face, i)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # This command let's us quit with the "q" button on a keyboard.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
