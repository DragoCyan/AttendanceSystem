import cv2
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') #pre-trained cascade model

cap = cv2.VideoCapture(0) # set to 1 for DroidCam Client

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # decreasing scaleFactor slows execution but raises precision; base val: 1.1
    # increasing minNeighbours makes the face check more strict; base val: 5
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # green rectangle of thickness 2
        cv2.imshow('Webcam Face Detection', frame)

    # Press Q to stop application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()