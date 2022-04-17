import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_image(path, shape=(224, 224), normalize=True):
    img = cv2.resize(cv2.imread(path), shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)

    if normalize:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    
    return img

class NST:
    def __init__(self):
        self.optimizer = keras.optimizers.Adam(lr=0.02, beta_1=0.99, epsilon=1e-1)
        self.vgg_model = NST.load_model()
        pass


    def loss_fn(self, style_outputs, content_outputs, style_target, content_target, content_weight, style_weight):
        content_loss = tf.reduce_mean((content_outputs - content_target)**2)
        style_loss = tf.add_n([tf.reduce_mean((output_ - target_)**2) for output_, target_ in zip(style_outputs, style_target)])
        total_loss = content_weight * content_loss + style_weight * style_loss
        return total_loss


    def load_model(input_shape=(224, 224, 3)):
        vgg = keras.applications.vgg19.VGG19(include_top=True, weights=None, input_shape=input_shape)
        vgg.load_weights("./models/VGG19/weights.h5")
        vgg.trainable = False
        
        content_layers = ["block4_conv2"]
        style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

        content_output = vgg.get_layer(content_layers[0]).output
        style_output = [vgg.get_layer(name).output for name in style_layers]
        gram_style_output = [NST.gram_matrix(_output) for _output in style_output]

        model = keras.Model([vgg.input], [content_output, gram_style_output])
        return model


    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        gram_matrix = tf.expand_dims(result, axis=0)
        input_shape = tf.shape(input_tensor)
        i_j = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return gram_matrix/i_j 


    def train_step(self, image, epoch, content_target, style_target, content_weight=1e-6, style_weight=1e-2):
        with tf.GradientTape() as tape:
            output = self.vgg_model([image*255])
            loss = self.loss_fn(output[1], output[0], style_target, content_target, content_weight, style_weight)

        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])

        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

        if epoch % 100 == 0:
            tf.print(f"Epoch {epoch} | Loss: {loss}")


    def style_image(self, content_path, style_path, epochs=1000, style_weight=1e-6, content_weight=1e-2):
        content_image = load_image(content_path)
        style_image = load_image(style_path)

        content_target = self.vgg_model([content_image*255])[0]
        style_target = self.vgg_model([style_image*255])[1]

        image = tf.image.convert_image_dtype(content_image, tf.float32)
        image = tf.Variable(image)
        
        for i in range(epochs):
            self.train_step(image, i, content_target, style_target, content_weight, style_weight)
        plt.imshow(image[0])
        plt.show()


class DCGAN:
    def __init__(self, weights, square_dims=128, latent_dim=128):
        self.generator = keras.model.load_model(weights)
        self.square_dims = square_dims
        self.latent_dim = latent_dim
        self.latent_size = int(square_dims/8)

    def generate(self, num_example=1):
        random_vector = tf.random.normal(shape=[num_example, self.latent_dim])
        generated_images = self.generator(random_vector)
        generated_images += 255
        generated_images = tf.clip_by_value(generated_images, clip_value_min=0.0, clip_value_max=255.0)
        generated_images = tf.image.convert_image_dtype(generated_images, tf.uint8)
        return generated_images


class SRGAN:
    def __init__(self, weights, lr_dim=64, hr_dim=256):
        self.generator = keras.model.load_model(weights)
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.scale_factor = hr_dim//lr_dim

    
    