import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
def get_ensemble(models, input_shape, weights=None):
    x = tf.keras.layers.Input(shape=input_shape)
    if weights is None:
        weights = [np.ones((1, 5)) / len(models) for _ in range(len(models))]

    y = [model(x) * w for model, w in zip(models, weights)]
    y = tf.reduce_sum(y, axis=0)
    model = tf.keras.Model(inputs=x, outputs=y)
    return model
def get_data():
    noaug_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        dtype=np.float32
    )
    train_generator = noaug_datagen.flow_from_directory(
        "../dataset/train/",
        class_mode="categorical",
        target_size=(224, 224),
        shuffle=False,
    )
    valid_generator = noaug_datagen.flow_from_directory(
        "../dataset/val/",
        class_mode="categorical",
        target_size=(224, 224),
        shuffle=False,
    )
    test_generator = noaug_datagen.flow_from_directory(
        "../dataset/test/",
        class_mode="categorical",
        target_size=(224, 224),
        shuffle=False,
    )
    return train_generator, valid_generator, test_generator
def get_metrics(model, data, name=None, show_results=True):
    if name is None:
        name = model.name
    y_true = data.labels
    y_pred = np.argmax(model.predict(data), axis=1)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracies = np.diag(cm)
    for i, a in enumerate(accuracies):
        report[str(i)]["accuracy"] = a
    if show_results:
        np.set_printoptions(precision=2)
        print(f"Accuracy Score - {name}: {acc}")
        print(f"Balanced Accuracy Score - {name}: {bal_acc}")
        print("\n")
        print(classification_report(y_true, y_pred))
        print("Confusion matrix:")
        print(cm)
        print("Classes accuracies", accuracies)
        np.set_printoptions(precision=None)
    return report
def compute_confusion_matrix(
    model,
    class_names,
    data,
    name=None,
):
    y_true = data.labels
    y_pred = np.argmax(model.predict(data), axis=1)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show(block=False)
def embed_preproc(model, preproc, input_shape):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Lambda(preproc, input_shape=input_shape),
            tf.keras.models.load_model(model),
        ]
    )
def load_models(models, input_shape):
    return [
        embed_preproc(model, preproc, input_shape) for model, preproc in models
    ]
class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
model_name = "Ensemble"
train, valid, test = get_data()
input_shape = 224, 224, 3
models = [
    [
        "models/model_ResNet50_ft.hdf5",
        tf.keras.applications.resnet50.preprocess_input,
    ],
    [
        "models/model_Xception_ft.hdf5",
        tf.keras.applications.xception.preprocess_input,
    ],
    [
        "models/model_Inception_ResNet_V2_ft.hdf5",
        tf.keras.applications.inception_resnet_v2.preprocess_input,
    ],
]

models = load_models(models, input_shape)
def get_ensemble_weights(models, metric):
    metrics = []
    for model in models:
        report = get_metrics(model, data=train, show_results=False)
        metrics.append([report[str(i)][metric] for i in range(5)])

    metrics = np.array(metrics)
    model_weights = metrics / np.sum(metrics, axis=0)
    return model_weights
ensemble_mean = get_ensemble(
    models,
    input_shape=(224, 224, 3),
)
ensemble_acc = get_ensemble(
    models,
    input_shape=(224, 224, 3),
    weights=get_ensemble_weights(models, "accuracy"),
)
ensemble_f1 = get_ensemble(
    models,
    input_shape=(224, 224, 3),
    weights=get_ensemble_weights(models, "f1-score"),
)
ensemble_mean.save("models/ensemble_mean.h5")
ensemble_acc.save("models/eemblensemble_acc.h5")
ensemble_f1.save("models/ens_f1.h5")
get_metrics(ensemble_mean, data=valid, name="ensemble_mean")
compute_confusion_matrix(
    ensemble_mean, class_names, data=valid, name="ensemble_mean"
)
get_metrics(ensemble_acc, data=valid, name="ensemble_acc")
compute_confusion_matrix(
    ensemble_acc, class_names, data=valid, name="ensemble_acc"
)
get_metrics(ensemble_f1, data=valid, name="ensemble_f1")
compute_confusion_matrix(
    ensemble_f1, class_names, data=valid, name="ensemble_f1"
)
get_metrics(ensemble_f1, data=test, name="ensemble_f1")
compute_confusion_matrix(
    ensemble_f1, class_names, data=test, name="ensemble_f1"
)
y_true =test.labels
y_pred = np.argmax(ensemble_f1.predict(test), axis=1)
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt=".1f",
    cmap="Purples",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title(f"Confusion Matrix - ensemble_f1")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show(block=False)

