import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

model=load_model('Model2.h5')

class_labels=['angry','disgust','fear','happy','neutral','sad','surprise']

cap=cv.VideoCapture(0)

face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
     ret,frame=cap.read()

     if not ret:
          break

     gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
     faces=face_cascade.detectMultiScale(gray,1.3,5)

     for (x,y,w,h) in faces:
          face = frame[y:y+h, x:x+w]
          face = cv.resize(face, (150, 150))
          face = face.astype("float") / 255.0
          face = img_to_array(face)
          face = np.expand_dims(face, axis=0) 

          preds = model.predict(face)[0]
          label = class_labels[np.argmax(preds)]
          confidence = np.max(preds)

          if confidence > 0.5:
               label_display = f"{label} ({confidence:.2f})"
          else:
               label_display = "Neutral"

          # Draw rectangle and label
          cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
          cv.putText(frame, 
                    f"{label} ({confidence:.2f})", 
                    (x, y-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

     cv.imshow("Facial Expression Recognition ", frame)

     if cv.waitKey(10) & 0xFF ==ord('a'):
          break

cap.release()
cv.destroyAllWindows()