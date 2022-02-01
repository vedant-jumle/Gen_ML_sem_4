import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2
import PIL

# configure the GPU
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class StyleTransfer:
    def __init__(self, vgg_weights="./models/VGG19/weights.h5", optimizer_name=keras.optimizers.Adam):
        self.vgg_weights = vgg_weights
        self.vgg_model = self.load_model()
        self.optimizer = optimizer_name(lr=0.02, beta_1=0.99, epsilon=1e-1)
    
    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        gram_matrix = tf.expand_dims(result, axis=0)
        input_shape = tf.shape(input_tensor)
        i_j = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return gram_matrix/i_j 

    def load_model(self, model_path, input_shape=(224, 224, 3)):
        vgg = keras.applications.vgg19.VGG19(include_top=True, weights=None, input_shape=input_shape)
        vgg.load_weights(model_path)
        vgg.trainable = False
        
        content_layers = ["block4_conv2"]
        style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

        content_output = vgg.get_layer(content_layers[0]).output
        style_output = [vgg.get_layer(name).output for name in style_layers]
        gram_style_output = [self.gram_matrix(_output) for _output in style_output]

        model = keras.Model([vgg.input], [content_output, gram_style_output])
        return model
    
    def loss_fn(self, style_outputs, content_outputs, style_target, content_target, content_weight, style_weight):
        content_loss = tf.reduce_mean((content_outputs - content_target)**2)
        style_loss = tf.add_n([tf.reduce_mean((output_ - target_)**2) for output_, target_ in zip(style_outputs, style_target)])
        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss
    
    def train_step(self, image, epoch, content_target, style_target, content_weight=1e-6, style_weight=1e-2):
        with tf.GradientTape() as tape:
            output = self.vgg_model([image*255])
            loss = self.loss_fn(output[1], output[0], style_target, content_target, content_weight, style_weight)

        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])

        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

        if epoch % 100 == 0:
            tf.print(f"Epoch {epoch} | Loss: {loss}")

def load_image(path, shape=(224, 224), normalize=True):
    img = cv2.resize(cv2.imread(path), shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)

    if normalize:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    
    return img