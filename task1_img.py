import cv2 as cv

img=cv.imread(r"C:\Users\Aditya Kumar Singh\OneDrive\Pictures\Camera Roll\WIN_20211219_21_18_41_Pro_1.jpg")

#Load cascade & detect face
def detectfaces(img):
     gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

     face_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
     faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

     for (x,y,w,h) in faces:
          cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

     cv.imshow('Faces on image',img)

if img is None:
     print("Can't open the file")
     exit()

while True:
     detectfaces(img)
     if cv.waitKey(0)==ord('a'):
          break

cv.destroyAllWindows()