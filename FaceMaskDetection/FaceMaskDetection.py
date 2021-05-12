import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import \
    ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

batch_size = 8
epochs = 30

directory = 'data'

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(70, 70),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='binary',
    seed=2020,
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    directory,
    target_size=(70, 70),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='binary',
    subset='validation')

imgs, labels = next(train_generator)


def plotImages(images_arr):
    plt.figure(figsize=(12, 12))
    i = 0
    for img in images_arr:
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title("with mask" if labels[i] == 0 else "without mask", fontsize=30)
        plt.axis("off")
        i += 1
    plt.show()


plotImages(imgs)
print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(70, 70, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=64, activation='relu'),
    Dense(units=1, activation='sigmoid'),
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochstoplot = range(1, epochs + 1)
plt.plot(epochstoplot, loss_train, 'g', label='Training loss')
plt.plot(epochstoplot, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accur_train = history.history['accuracy']
accur_val = history.history['val_accuracy']
plt.plot(epochstoplot, accur_train, 'g', label='Training accuracy')
plt.plot(epochstoplot, accur_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from tkinter import Label, Tk, Button
from PIL import Image, ImageTk
import tkinter.filedialog


def choose():
    path = tkinter.filedialog.askopenfilename(filetypes=[("Image File", '.jpg')])
    im = Image.open(path)
    im = im.resize((200, 200), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(im)
    myvar = Label(root, image=tkimage)
    myvar.image = tkimage
    myvar.pack()

    img_pred = image.load_img(path, target_size=(70, 70))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)

    prediction = model.predict(img_pred)

    if int(prediction[0][0]) == 0:
        result = "The person is wearing a mask."
    else:
        result = "The person is not wearing a mask."

    text = Label(root, text=result)
    text.pack()


root = Tk()
root.geometry("1000x1000")
Label(root, text="   ").pack()
Label(root, text="choose photo").pack()
Label(root, text="   ").pack()
Button(text="Choose", command=choose).pack()
Label(root, text="   ").pack()
root.mainloop()
