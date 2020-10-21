import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import preprocessing
from utils import *


#################### LOAD DATA

labels_df = pd.read_csv(LABELS_PATH)

class_names = labels_df["Name"].to_numpy()
# print(class_names)

# 34799 files --> 1088 batches
signs_ds = preprocessing.image_dataset_from_directory(IMAGES_PATH,
                                                      color_mode="grayscale",
                                                      image_size=(IMG_HEIGHT, IMG_WEIGHT),
                                                      seed=SEED)

# val_ds = preprocessing.image_dataset_from_directory(images_path,
#                                                     color_mode="grayscale",
#                                                     image_size=(img_height, img_width),
#                                                     seed=seed,
#                                                     validation_split=0.2,
#                                                     subset="validation")

signs_ds = signs_ds.shuffle(999999)
train_ds = signs_ds.take(int(1088 * 0.7))
test_ds = signs_ds.skip(int(1088 * 0.7))
val_ds = test_ds.skip(int(1088 * 0.15))
test_ds = test_ds.take(int(1088 * 0.15))

#################### VISUALIZE DATA

plt.figure(figsize=(7, 7))
for signs, labels in train_ds.take(1):
    for i in range(len(signs)):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(tf.squeeze(signs[i].numpy().astype("uint8")), cmap="gray")
        plt.title(class_names[labels[i]], fontsize=5)
        plt.axis("off")

plt.savefig("./Trainer_Output/training_examples.png")

#################### CONFIGURE THE DATASET FOR PERFORMANCE

AUTOTUNE = tf.data.experimental.AUTOTUNE

'''
.cache() --> Keeps the images in memory after they're loaded off disk during the first epoch.
.prefetch() --> Overlaps data preprocessing and model execution while training.
'''

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#################### VISUALIZE DATA AUGMENTATION (EXAMPLE)

plt.figure(figsize=(5, 5))
for signs, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(signs)  # Pass images (32) in the batch through the layer (This is done 9 times)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(tf.squeeze(augmented_images[0].numpy().astype("uint8")), cmap="gray")  # Plot only the first one, to the see the effects of data augmentation
        plt.axis("off")

plt.savefig("./Trainer_Output/data_augmentation_ex.png")

#################### CREATE, TRAIN and SAVE THE MODEL

model = create_model(class_names)

print_clr("-------------------- TRAINING MODEL --------------------", MAGENTA)
history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS)

print_clr("------------------- MODEL EVALUATION -------------------", MAGENTA)
test_loss, test_acc = model.evaluate(test_ds)
print_clr(f"Test Loss: {test_loss}", GREEN)
print_clr(f"Test Accuracy: {test_acc}", GREEN)

print_clr("--------------------- SAVING MODEL ---------------------", MAGENTA)
model.save_weights("./Checkpoints/weights")

#################### VISUALIZE TRAINING RESULTS

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUM_EPOCHS)

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

plt.savefig("./Trainer_Output/training_results.png")
