import cv2 as cv

cap=cv.VideoCapture(0)

#Define codec & create the object
fourcc=cv.VideoWriter_fourcc(*'mp4v')
out=cv.VideoWriter("vid1.mp4",fourcc,20,(500,500))

if not cap.isOpened():
     print("Can't open the camera")
     exit()

while True:
     #Capture frame-by-frame
     ret,frame=cap.read()

     #if frame is read, ret=true else false
     if not ret:
          print("Can't receive frame")
          break

     #Fliping the frame vertically
     #frame=cv.flip(frame,0)
     
     frame_res=cv.resize(frame,(500,500))
     out.write(frame_res)

     #gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
     cv.imshow('frame',frame)
     if cv.waitKey(1)==ord('a'):
          break

cap.release()
out.release()
cv.destroyAllWindows()