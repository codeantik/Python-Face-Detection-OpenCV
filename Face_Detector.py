from cv2 import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# img = cv2.imread('people.jpg')
webcam = cv2.VideoCapture(0)

while True:

    successfull_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates: 
     cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Python Face Detector', frame)

    key = cv2.waitKey(1)

    if key == 83 or key == 113:
        break

webcam.release()


print("Completed")
# print(face_coordinates)
# (x, y, w, h) = face_coordinates[0]
# for (x, y, w, h) in face_coordinates: 
    # cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)



# cv2.imshow('Python Face Detector', img)

# cv2.waitKey()
