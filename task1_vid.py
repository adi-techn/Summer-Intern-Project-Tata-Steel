import cv2 as cv

#Load cascade & detect faces
face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
     print("Empty cascade")
     exit()

vid=cv.VideoCapture(0)
if not vid.isOpened():
     print("Can't open the file")
     exit()

while True:
     ret,frame=vid.read() 
     if not ret:
          print("Can't read the frames")
          break

     gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
     
     for (x,y,w,h) in faces:
          cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

     cv.imshow("Video",frame)
     if cv.waitKey(1) & 0xFF == ord('a'):
          break

vid.release()
cv.destroyAllWindows()

