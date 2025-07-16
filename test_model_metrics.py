from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

model=load_model('Model2.h5')

test=ImageDataGenerator(rescale=1/255)
test_dataset=test.flow_from_directory(r"C:\Users\Aditya Kumar Singh\Downloads\archive\images\images\test",
                                      target_size=(150,150),
                                      batch_size=32,
                                      class_mode='categorical',
                                      shuffle=False)

pred=model.predict(test_dataset)
y_pred=np.argmax(pred,axis=1)
y_true=test_dataset.classes

cm=confusion_matrix(y_true,y_pred)
acc=accuracy_score(y_true,y_pred)
prec=precision_score(y_true,y_pred,average='weighted')
rec=recall_score(y_true,y_pred,average='weighted')
f1=f1_score(y_true,y_pred,average='weighted')

print("Confusion Matrix :\n",cm)
print("Accuracy : ",acc)
print("Precision : ",prec)
print("Recall : ",rec)
print("F1 Score : ",f1)