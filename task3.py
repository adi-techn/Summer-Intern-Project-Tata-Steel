import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train=ImageDataGenerator(rescale=1/255)
test=ImageDataGenerator(rescale=1/255)

train_dataset=train.flow_from_directory(r"C:\Users\Aditya Kumar Singh\Downloads\archive\images\images\train",
                                        target_size=(150,150),
                                        batch_size=32,
                                        class_mode='categorical')

test_dataset=test.flow_from_directory(r"C:\Users\Aditya Kumar Singh\Downloads\archive\images\images\test",
                                      target_size=(150,150),
                                      batch_size=32,
                                      class_mode='categorical')
print(train_dataset.class_indices)

model=keras.Sequential([keras.layers.Input(shape=(150,150,3))])

model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D(2,2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))

model.add(keras.layers.Dense(train_dataset.num_classes,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_dataset,epochs=10,validation_data=test_dataset)

model.save('Model2.h5')