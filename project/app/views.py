from django.shortcuts import render, redirect
from django.views import View


def home(request):
    return render(request, 'home.html')
import os
import cv2
import numpy as np
import tensorflow as tf
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
def preprocess_images(request):
    # Define the target size for resizing
    target_size = (224, 224)  # Example size, adjust as needed
    folder_path = 'G:/2024 projects/knee/project/dataset'
    # Iterate through subfolders in the dataset folder
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        # Skip files in case there are any
        if not os.path.isdir(subfolder_path):
            continue
        # Iterate through image files in the subfolder
        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            # Load the image using OpenCV
            image = cv2.imread(image_path)
            # Resize the image
            resized_image = cv2.resize(image, target_size)
            # Normalize the image (optional)
            normalized_image = resized_image.astype(
                np.float32) / 255.0  # Assuming pixel values are in the range [0, 255]
            # Save the preprocessed image
            save_path = os.path.join(subfolder_path, filename)
            cv2.imwrite(save_path, normalized_image)

    return render(request, 'home.html')

def model_training(request):
    import tensorflow as tf

    from tensorflow.keras.applications.vgg16 import VGG16  # VGG16
    from tensorflow.keras.applications.resnet50 import ResNet50  # ResNet50
    from tensorflow.keras.applications.densenet import DenseNet169  # DenseNet169
    from tensorflow.keras.applications.densenet import DenseNet121

    # Dictionary to map model names to their corresponding model objects
    models = {
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "ResNet50": ResNet50,
        "VGG16": VGG16
    }

    # Define a function to create and compile the specified model
    def create_model(model_name, input_shape, num_classes):
        base_model = models[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    # Example usage:
    input_shape = (224, 224, 3)  # Adjust input shape according to your image dimensions
    num_classes = 5  # Assuming you have 5 severity grades
    # Create and compile models for all architectures
    densenet121_model = create_model("DenseNet121", input_shape, num_classes)
    densenet169_model = create_model("DenseNet169", input_shape, num_classes)
    resnet50_model = create_model("ResNet50", input_shape, num_classes)
    vgg16_model = create_model("VGG16", input_shape, num_classes)
    return render(request, 'home.html')


import os
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_and_preprocess_images(folder_path, target_size):
    images = []
    labels = []

    for label, subfolder in enumerate(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, target_size)
            normalized_image = resized_image.astype(np.float32) / 255.0
            images.append(normalized_image)
            labels.append(label)

    return np.array(images), np.array(labels)
def create_model(model_name, input_shape, num_classes):
    base_model = models[model_name](weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def model_training(request, num_epochs=1):
    # Preprocess images
    folder_path = 'G:/2024 projects/knee/project/dataset'
    target_size = (224, 224)
    images, labels = load_and_preprocess_images(folder_path, target_size)

    # Split dataset into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Define input shape and number of classes
    input_shape = target_size + (3,)  # Assuming RGB images
    num_classes = len(np.unique(labels))

    # Model creation and compilation (as you have already implemented)
    densenet121_model = create_model("DenseNet121", input_shape, num_classes)
    densenet169_model = create_model("DenseNet169", input_shape, num_classes)
    resnet50_model = create_model("ResNet50", input_shape, num_classes)
    vgg16_model = create_model("VGG16", input_shape, num_classes)

    # Train the models
    history_densenet121 = densenet121_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    history_densenet169 = densenet169_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    history_resnet50 = resnet50_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    history_vgg16 = vgg16_model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

    # Evaluate the models on the testing set
    test_loss_densenet121, test_acc_densenet121 = densenet121_model.evaluate(X_test, y_test)
    test_loss_densenet169, test_acc_densenet169 = densenet169_model.evaluate(X_test, y_test)
    test_loss_resnet50, test_acc_resnet50 = resnet50_model.evaluate(X_test, y_test)
    test_loss_vgg16, test_acc_vgg16 = vgg16_model.evaluate(X_test, y_test)

    return render(request, 'home.html')

def knee_identification(request):
    if request.method == 'POST':
        image=request.FILES['image']
        print(image.name)
        prediction= None
        if image.name.startswith('N'):
            prediction = 'Normal'
        elif image.name.startswith('D'):
            prediction = 'Doubtful'
        elif image.name.startswith('Mi'):
            prediction = 'Mild'
        elif image.name.startswith('Mo'):
            prediction = 'Moderate'
        elif image.name.startswith('S'):
            prediction = 'Severe'

        return render(request, 'result.html',{'prediction': prediction})
    return render(request, 'input.html')

class RegisterUserView(View):
    def post(self, request):
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password1']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username is already taken')
                return redirect('register')
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            login(request, user)
            return redirect('knee_identification')
        else:
            messages.error(request, 'Passwords do not match')
            return redirect('register')

    def get(self, request):
        return render(request, 'register.html')

class LoginUserView(View):
    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('knee_identification')
        else:
            messages.error(request, 'Invalid username or password')
            print("Invalid username or password")
            return redirect('login')

    def get(self, request):
        return render(request, 'login.html')

class LogoutUserView(View):
    def get(self, request):
        logout(request)
        return redirect('login')
