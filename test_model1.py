from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

model =load_model('Model1.h5')

test=ImageDataGenerator(rescale=1/255)
test_dataset = test.flow_from_directory(
    r"C:\Users\Aditya Kumar Singh\Downloads\kagglecatsanddogs_5340\CatsDogsSplit\test",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    shuffle=False   # VERY IMPORTANT
)

pred_prob=model.predict(test_dataset)
y_true=test_dataset.classes
y_pred=np.argmax(pred_prob,axis=1)

def predictImg(file):
     img1=image.load_img(file,target_size=(150,150))
     plt.imshow(img1)

     Y=image.img_to_array(img1)
     X=np.expand_dims(Y,axis=0)
     X=X/255
     val=model.predict(X)
     print("Raw Prediction : ",val)

     if val[0][0]>=0.5:
          plt.title("Dog",fontsize=30)
     else :
          plt.title("Cat",fontsize=30)

     plt.show()

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

predictImg(r"C:\Users\Aditya Kumar Singh\Downloads\dog.webp")