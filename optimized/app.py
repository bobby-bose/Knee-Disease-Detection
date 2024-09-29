import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * 0.5  # Example scaling operation

    @classmethod
    def from_config(cls, config):
        config.pop('scale', None)  # Remove unexpected 'scale' argument
        return cls(**config)

# Register custom layer
tf.keras.utils.get_custom_objects()["CustomScaleLayer"] = CustomScaleLayer

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def main():
    class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
    model = tf.keras.models.load_model("./models/model.hdf5", custom_objects={'CustomScaleLayer': CustomScaleLayer})
    target_size = (224, 224)
    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[
            grad_model.get_layer("global_average_pooling2d_1").input,
            grad_model.output,
        ],
    )
    uploaded_file = "test.png"  # Update with the path to your image
    if uploaded_file is not None:
        img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_aux = img.copy()
        img_array = np.expand_dims(img_aux, axis=0)
        img_array = np.float32(img_array)
        img_array = tf.keras.applications.xception.preprocess_input(img_array)
        y_pred = model.predict(img_array)
        y_pred = 100 * y_pred[0]
        number = np.argmax(y_pred)
        grade = class_names[number]
        print(f"Predicted Severity Grade: {grade} - {np.amax(y_pred):.2f}%")
        heatmap = make_gradcam_heatmap(grad_model, img_array)
        image = save_and_display_gradcam(img, heatmap)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        print("Prediction Visualization saved as 'gradcam_visualization.png'")

if __name__ == "__main__":
    main()
