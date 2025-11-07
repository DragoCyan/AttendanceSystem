import cv2
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') #pre-trained cascade model
eyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # decreasing scaleFactor slows execution but raises precision; base val: 1.1
    # increasing minNeighbours makes the face check more strict; base val: 5
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # green rectangle of thickness 2

        roiGray = gray[y:y+h, x:x+w]
        roiColor = frame[y:y+h, x:x+w]

        eyes = eyeCascade.detectMultiScale(roiGray, 1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frame, 'Eyes Detected', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        smile = smileCascade.detectMultiScale(roiGray, 1.7, 20)
        if len(smile) > 0:
            cv2.putText(frame, 'Smile Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Smart Face Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()