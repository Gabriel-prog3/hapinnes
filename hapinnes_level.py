# import required libraries
import cv2
import random
import numpy as np


logo = cv2.imread("Asset 4.png")

size = 100
logo = cv2.resize(logo, (size, size))
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 2, 255, cv2.THRESH_BINARY)


# read haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# read haar cascade for smile detection
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()

# Detects faces in the input image
faces = face_cascade.detectMultiScale(frame, 1.3, 5)

# loop over all the faces detected
happines = 2
happines2 = 0

vel = 5
counter = 0
max = 0


while True:
    counter += 1

    ret, frame = cap.read()

    roi = frame[-size - 10 : -10, -size - 10 : -10]
    roi[np.where(mask)] = 0
    roi += logo

    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        x, y, w, h = faces[0]
    except:
        x, y, w, h = 0, 0, 0, 0

    # draw a rectangle in a face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.rectangle(
        frame, (x + w // 2 - 15, y + 2), (x + w // 2 - 9, y + 20), (255, 255, 255), -1
    )
    cv2.rectangle(
        frame, (x + w // 2 - 2, y + 2), (x + w // 2 + 4, y + 20), (255, 255, 255), -1
    )
    cv2.rectangle(
        frame, (x + w // 2 + 11, y + 2), (x + w // 2 + 17, y + 20), (255, 255, 255), -1
    )

    cv2.putText(
        frame,
        str(happines) + "%",
        (x + w + 15, y + 100),
        cv2.FONT_HERSHEY_TRIPLEX,
        1.4,
        (255, 255, 255),
        4,
    )

    cv2.putText(
        frame,
        "^" + str(happines2) + "%",
        (70, 420),
        cv2.FONT_HERSHEY_TRIPLEX,
        1.5,
        (255, 255, 255),
        4,
    )
    cv2.putText(
        frame,
        "mejor momento",
        (20, 460),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.8,
        (255, 255, 255),
        1,
    )
    roi_gray = gray[y : y + h, x : x + w]
    roi_color = frame[y : y + h, x : x + w]

    # if counter % 150 == 0:
    # cv2.imwrite(str(counter) + ".jpg", frame)

    # detecting smile within the face roi
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    if len(smiles) > 0:
        happines += vel
        happines2 = happines
    else:
        happines -= vel + 4

    if happines <= 3:
        happines = 0

    if counter % 30 == 0:
        max = random.randint(4, 38)

    if happines >= max:
        happines = max
        happines2 = happines

    cv2.imshow("nFrame", frame)
    k = cv2.waitKey(20)
    if k == 27:
        break

# Display an image in a window
# cv2.imshow("Smile Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
