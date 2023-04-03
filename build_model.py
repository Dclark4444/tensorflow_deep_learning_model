#Import used libraries
import tensorflow, os, cv2, imghdr
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib as plt

#Controls use of GPU
gpus = tensorflow.config.experimental.list_physical_devices("GPU")
for current_gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(current_gpu, True)

#File path to get to the raw train data
train_path = os.path.join("raw_data", "afhq", "train")

#Pipeline to feed in data set
raw_data = tensorflow.keras.utils.image_dataset_from_directory(train_path)
data_iterator = raw_data.as_numpy_iterator()
batch = data_iterator.next() #Using default batch settings

scaled_data = raw_data.map(lambda x,y: (x/255, y))

#Making training and testing data sets
train_size = int(len(scaled_data)*.7)
val_size = int(len(scaled_data)*.2)
test_size = int(len(scaled_data)*.1)

train = scaled_data.take(train_size)
val = scaled_data.skip(train_size).take(val_size)
test = scaled_data.skip(train_size+val_size).take(test_size)

#Creating learning model
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tensorflow.losses.BinaryCrossentropy(), metrics=['accuracy'])

#Training and logging the model's progress
logdir='logs'
tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Graphing the models training
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.legend(loc="upper left")
plt.savefig(os.path.join("saved_graphs", "graphed_model.png"))

#Saving the trained model
model.save(os.path.join("saved_models", "designedModel.h5"))


















