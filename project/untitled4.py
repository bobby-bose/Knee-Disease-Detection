import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16, DenseNet121, DenseNet169
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import cv2
import os

# Load and preprocess X-ray image dataset
def load_dataset(train_dataset_path, test_dataset_path, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    test_generator = test_datagen.flow_from_directory(
        test_dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator

# Create model with specified base model
def create_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Combine predictions using simple average method
def ensemble_predictions(models, data):
    ensemble_preds = []
    for model in models:
        preds = model.predict(data)
        ensemble_preds.append(preds)
    return np.mean(ensemble_preds, axis=0)

# Visualize using Eigen CAM
from tensorflow.keras.layers import Layer

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class GradCAM(Layer):
    def __init__(self, model, class_idx):
        super(GradCAM, self).__init__()
        self.model = model
        self.class_idx = class_idx

    def call(self, inputs):
        last_conv_layer = self.model.get_layer(index=-3)
        class_output = self.model.output[:, self.class_idx]
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([inputs])
        for i in range(conv_layer_output_value.shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

def eigen_cam(model, img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)
    gradcam = GradCAM(model, 0)  # Assuming we want to visualize the activation of the first class
    heatmap = gradcam(img)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img[0], 0.5, heatmap, 0.5, 0)
    return superimposed_img


# Constants
TRAIN_DATASET_PATH = 'G:/2024 projects/knee/project/dataset'
TEST_DATASET_PATH = 'G:/2024 projects/knee/project/test'
NUM_CLASSES = 5  # Assuming there are 5 classes
IMG_SIZE = (224, 224)

# Load dataset
train_generator, validation_generator, test_generator = load_dataset(TRAIN_DATASET_PATH, TEST_DATASET_PATH, target_size=IMG_SIZE)

# Create base models
base_models = [
    ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    # VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    # DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    # DenseNet169(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
]

# Create models using base models
models = [create_model(base_model) for base_model in base_models]

# Compile models
for model in models:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train models (optional)
for model in models:

        model.fit(
            train_generator,
            epochs=1,
            steps_per_epoch=len(train_generator),
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
        )

# Combine predictions using simple average method
import numpy as np

# Combine predictions using simple average method
test_generator.reset()  # Reset the test generator to start from the beginning
ensemble_preds = []

for _ in range(len(test_generator)):
    test_data, _ = test_generator.next()
    ensemble_preds.append(ensemble_predictions(models, test_data))

ensemble_preds = np.mean(ensemble_preds, axis=0)

# Handling runtime warning related to invalid values
with np.errstate(invalid='ignore'):
    print("Ensemble Predictions:", ensemble_preds)

# Visualize ensemble output using Eigen CAM (for the first image in test dataset)
first_test_image_path = os.path.join(TEST_DATASET_PATH, os.listdir(TEST_DATASET_PATH)[0])
ensemble_cam_img = eigen_cam(models[0], first_test_image_path)

# Display results
print("Ensemble Predictions:", ensemble_preds)
cv2.imshow('Ensemble CAM', ensemble_cam_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



