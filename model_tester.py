import pandas as pd
import numpy as np

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

#################### PREDICT ON NEW/UNSEEN DATA

loaded_model = create_model(class_names)
loaded_model.load_weights("./Checkpoints/weights")

loaded_model.summary()

# img = preprocessing.image.load_img("./Resources/Test-examples/stop_ex.jpeg", color_mode="grayscale", target_size=(img_height, img_width))
# img_arr = preprocessing.image.img_to_array(img)
#
# plt.imshow(img_arr.reshape((32,32)), cmap="gray")
# plt.savefig("./Resources/Test-examples/stop_ex_preprocessed.png")
#
# img_arr = np.array([img_arr])  # Convert single image to a batch. ALTERNATIVE --> tf.expand_dims(img, 0)
# img_arr = img_arr * (1./255)  # Normalize
#
# predictions = model.predict(img_arr)
# score = tf.nn.softmax(predictions[0])
#
# print(score)
#
# print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.")

print_clr("------------------- MODEL EVALUATION -------------------", MAGENTA)
test_loss, test_acc = loaded_model.evaluate(test_ds)
print_clr(f"Test Loss: {test_loss}", GREEN)
print_clr(f"Test Accuracy: {test_acc}", GREEN)
