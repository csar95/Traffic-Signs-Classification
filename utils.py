from tensorflow.keras import models, layers, losses


IMAGES_PATH = "./Resources/Traffic-signs"
LABELS_PATH = "./Resources/labels.csv"
IMG_HEIGHT, IMG_WEIGHT = 32, 32
SEED = 42
NUM_EPOCHS = 10

RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
MAGENTA = "\x1b[35m"
CYAN = "\x1b[36m"
GREY = "\x1b[90m"
RESET = "\x1b[0m"


def print_clr(msg, color):
    print(color + msg + RESET)

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
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WEIGHT, 1)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

#################### DL MODEL CREATION

def create_model(class_names):
    new_model = models.Sequential([
        data_augmentation,  # Transforms the input images according to the functions inside this layer. It's a form of regularization
        normalization_layer,  # Normalizes the image so that the pixel values are in range [0, 1]
        layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),  # same --> zero-padding | valid --> no-padding
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

    new_model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),  # Generally used in multi-class classification problems
                  metrics=['accuracy'])

    return new_model
