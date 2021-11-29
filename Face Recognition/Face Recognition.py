import cv2
import numpy as np
import face_recognition
import os



path = 'images'
Images = []
classname = []
List = os.listdir(path)
print(List)

for c1 in List:
    curing = cv2.imread(f'{path}/{c1}')
    Images.append(curing)
    classname.append(os.path.splitext(c1)[0])

def findEncode(Images):
    encodelist = []
    for img in Images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistKnow = findEncode(Images)
print("encoding complete")

cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesFrame = face_recognition.face_locations(imgs)
    encodeFrame = face_recognition.face_encodings(imgs, facesFrame)

    for encodeface, faceLoc in zip(encodeFrame, facesFrame):
        matches = face_recognition.compare_faces(encodelistKnow,encodeface)
        faceDistance = face_recognition.face_distance(encodelistKnow, encodeface)
        matchesIndex = np.argmin(faceDistance)

        if matches[matchesIndex]:
            name = classname[matchesIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 =   y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (250, 250, 250), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 230, 10),2)


    cv2.imshow('webcam', img)
    cv2.waitKey(1)

