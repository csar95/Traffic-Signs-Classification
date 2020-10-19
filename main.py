import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import models, layers, preprocessing


images_path = "./Resources/Traffic-signs"
labels_path = "./Resources/labels.csv"
img_height, img_width = 32, 32
seed = 42

#################### LOAD DATA

labels_df = pd.read_csv(labels_path)

class_names = labels_df["Name"].to_numpy()
# print(class_names)

train_ds = preprocessing.image_dataset_from_directory(images_path,
                                                    color_mode="grayscale",
                                                    image_size=(img_height, img_width),
                                                    seed=seed,
                                                    validation_split=0.2,
                                                    subset="training")

val_ds = preprocessing.image_dataset_from_directory(images_path,
                                                    color_mode="grayscale",
                                                    image_size=(img_height, img_width),
                                                    seed=seed,
                                                    validation_split=0.2,
                                                    subset="validation")

#################### VISUALIZE DATA

plt.figure(figsize=(7, 7))
for signs, labels in train_ds.take(1):
    for i in range(len(signs)):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(tf.squeeze(signs[i].numpy().astype("uint8")), cmap="gray")
        plt.title(class_names[labels[i]], fontsize=5)
        plt.axis("off")

plt.savefig("./Resources/Output_data/training_examples.png")

#################### CONFIGURE THE DATASET FOR PERFORMANCE

AUTOTUNE = tf.data.experimental.AUTOTUNE

'''
.cache() --> Keeps the images in memory after they're loaded off disk during the first epoch.
.prefetch() --> Overlaps data preprocessing and model execution while training.
'''

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#################### STANDARDIZE THE DATA

'''
Pixel values are in the [0, 255] range. This is not ideal for a neural network.
'''
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

#################### DATA AUGMENTATION

'''
This technique generates additional training data from the existing examples by augmenting then using random transformations
that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.
'''

data_augmentation = models.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 1)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

plt.figure(figsize=(5, 5))
for signs, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(signs)  # Pass images (32) in the batch through the layer (This is done 9 times)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(tf.squeeze(augmented_images[0].numpy().astype("uint8")), cmap="gray")  # Plot only the first one, to the see the effects of data augmentation
        plt.axis("off")

plt.savefig("./Resources/Output_data/data_augmentation_ex.png")

#################### CREATE, TRAIN and SAVE THE MODEL

model = models.Sequential([
    data_augmentation,  # Transforms the input images according to the functions inside this layer. It's a form of regularization
    normalization_layer,  # Normalizes the image so that the pixel values are in range [0, 1]
    layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),  # A technique to reduce overfitting. It's a form of regularization
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(class_names.size)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Generally used in multi-class classification problems
              metrics=['accuracy'])

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

model.save("trained_model")

#################### VISUALIZE TRAINING RESULTS

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, '-', label='Training Accuracy')
plt.plot(epochs_range, val_acc, '--', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, '-', label='Training Loss')
plt.plot(epochs_range, val_loss, '--', label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig("./Resources/Output_data/training_results.png")

#################### PREDICT ON NEW DATA

loaded_model = models.load_model("trained_model")
loaded_model.summary()

img = preprocessing.image.load_img("./Resources/Test-examples/stop_ex.jpeg", color_mode="grayscale", target_size=(img_height, img_width))
img_arr = preprocessing.image.img_to_array(img)

plt.imshow(img_arr.reshape((32,32)), cmap="gray")
plt.savefig("./Resources/Test-examples/stop_ex_preprocessed.png")

img_arr = np.array([img_arr])  # Convert single image to a batch. ALTERNATIVE --> tf.expand_dims(img, 0)
img_arr = img_arr * (1./255)  # Normalize

predictions = loaded_model.predict(img_arr)
score = tf.nn.softmax(predictions[0])

print(score)

print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.")

# TODO: TRY TO GENERATE TEST DATA SO WE CAN EVALUATE THE MODEL
