import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

model=load_model('Model2.h5')

class_labels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade=cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")

path=r"C:\Users\Aditya Kumar Singh\OneDrive\Pictures\Saved Pictures\sample.jpg"
img=cv.imread(path)
if img is None:
     print("Can't load file")

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
          face = img[y:y+h, x:x+w]
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
               label_display = "Uncertain"

          cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
          cv.putText(img, 
                    f"{label} ({confidence:.2f})", 
                    (x, y-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
          
cv.imshow("Facial Recognition Prediciton",img)
cv.waitKey(0)
cv.destroyAllWindows()