import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Definning model
model=keras.Sequential([keras.layers.Input(shape=(150,150,3))])

#Conv & Maxpool layer 1
model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

#Conv & Maxpool layer 2
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

#Conv & Maxpool layer 3
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))
     
#Image matrix to 1D array
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))

#Output layer
model.add(keras.layers.Dense(1,activation='sigmoid'))


train=ImageDataGenerator(rescale=1/255)
test=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory(r"C:\Users\Aditya Kumar Singh\Downloads\kagglecatsanddogs_5340\CatsDogsSplit\train",
                                        target_size=(150,150),
                                        batch_size=32,
                                        class_mode='binary')

test_dataset=test.flow_from_directory(r"C:\Users\Aditya Kumar Singh\Downloads\kagglecatsanddogs_5340\CatsDogsSplit\test",
                                      target_size=(150,150),
                                      batch_size=32,
                                      class_mode='binary',
                                      shuffle=False)

#Training model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Passing input dataset to model
#steps_per_epoch = training image batch size
model.fit(train_dataset,steps_per_epoch=250,epochs=10,validation_data=test_dataset)

#print(train_dataset.class_indices)
model.save("Model1.h5")