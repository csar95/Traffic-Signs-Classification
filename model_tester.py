import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import preprocessing
from utils import *


#################### LOAD DATA

labels_df = pd.read_csv(LABELS_PATH)

class_names_ori = labels_df["Name"].to_numpy()
# print(class_names_ori)

# 34799 files --> 1088 batches
signs_ds = preprocessing.image_dataset_from_directory(IMAGES_PATH,
                                                      color_mode="grayscale",
                                                      image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      seed=SEED)
class_names_ds = signs_ds.class_names
# print(class_names_ds)

signs_ds = signs_ds.shuffle(999999)
train_ds = signs_ds.take(int(1088 * 0.7))
test_ds = signs_ds.skip(int(1088 * 0.7))
val_ds = test_ds.skip(int(1088 * 0.15))
test_ds = test_ds.take(int(1088 * 0.15))

#################### LOAD TRAINED MODEL
print_clr("-------------------- LOADING MODEL ---------------------", MAGENTA)
model = create_model(class_names_ori)
model.load_weights("./Checkpoints/weights")
model.summary()

print_clr("------------------- MODEL EVALUATION -------------------", MAGENTA)
test_loss, test_acc = model.evaluate(test_ds)
print_clr(f"Test Loss: {test_loss}", GREEN)
print_clr(f"Test Accuracy: {test_acc}", GREEN)

#################### READ WEB-CAM
# cam = cv2.VideoCapture(0)  # 0 --> Default web-cam
# cam.set(propId=3, value=640)  # 3 --> Width
# cam.set(propId=4, value=480)  # 4 --> Height
# cam.set(propId=10, value=100)  # 10 --> Brightness
#
# while True:
#     _, imgOriginal = cam.read()
#
#     # IMAGE PREPROCESSING
#     imgResized = cv2.resize(imgOriginal, dsize=(IMG_WIDTH, IMG_HEIGHT))
#     imgGray = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)
#     imgNormalized = imgGray * (1. / 255)
#
#     cv2.imshow("Processed Image", imgNormalized)
#
#     cv2.putText(imgOriginal, text="CLASS: ", org=(20, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
#     cv2.putText(imgOriginal, text="PROBABILITY: ", org=(20, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
#
#     # PREDICT ON NEW/UNSEEN DATA
#     img_arr = imgNormalized.reshape(1, 32, 32, 1)  # Convert single image to a batch
#     predictions = model.predict(img_arr)
#     score = tf.nn.softmax(predictions[0])
#     prob_value = np.max(score)
#
#     if prob_value > PROB_THRESHOLD:
#         cv2.putText(imgOriginal, text=f"[{np.argmax(score)}] {class_names[np.argmax(score)]}", org=(120, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
#         cv2.putText(imgOriginal, text=f"{round(prob_value * 100, 2)}%", org=(190, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
#
#     cv2.imshow(winname="Web-cam input", mat=imgOriginal)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop the video
#         break

#################### PREDICT ON NEW/UNSEEN DATA

img = preprocessing.image.load_img("./Resources/Test-examples/stop_ex.jpeg", color_mode="grayscale", target_size=(IMG_HEIGHT, IMG_WIDTH))
img_arr = preprocessing.image.img_to_array(img)
# for signs, labels in train_ds.take(1):
#     img_arr = preprocessing.image.img_to_array(signs[0])
#     print(class_names_ori[ int(class_names_ds[labels[0]]) ])
#     break

plt.imshow(img_arr.reshape((32,32)), cmap="gray")
# plt.savefig("./Resources/Test-examples/stop_ex_preprocessed.png")
plt.show()

img_arr = np.array([img_arr])  # Convert single image to a batch. ALTERNATIVE --> tf.expand_dims(img, 0)
img_arr = img_arr * (1./255)  # Normalize

predictions = model.predict(img_arr)
score = tf.nn.softmax(predictions[0])

print(f"This image most likely belongs to {class_names_ori[ int(class_names_ds[np.argmax(score)]) ]} with a {100 * np.max(score)} percent confidence.")
