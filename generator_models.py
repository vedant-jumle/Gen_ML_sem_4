import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2
from PIL import Image

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

    def style_image(self, content_path, style_path, epochs=1000, style_weight=1e-6, content_weight=1e-2):
        content_image = load_image(content_path)
        style_image = load_image(style_path)

        content_target = self.vgg_model([content_image*255])[0]
        style_target = self.vgg_model([style_image*255])[1]

        image = tf.image.convert_image_dtype(content_image, tf.float32)
        image = tf.Variable(image)
        
        for i in range(epochs):
            self.train_step(image, i, content_target, style_target, content_weight, style_weight)
        
        # save the image
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.encode_jpeg(image)
        tf.io.write_file("./output/NST_result.png", image)

def load_image(path, shape=(224, 224), normalize=True):
    img = cv2.imread(path)
    if shape:
        img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)

    if normalize:
        img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

class DCGAN:
    def __init__(self, model_gen_path, latent_dim=128):
        self.model_gen_path = model_gen_path
        self.model = self.load_model()
        self.latent_dim = latent_dim

    def load_model(self):
        model = keras.models.load_model(self.model_gen_path)
        return model

    def generate(self):
        random_vector = tf.random.normal(shape=(1, self.latent_dim))
        generated_image = self.model(random_vector)[0]
        generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
        generated_image = tf.image.convert_image_dtype(generated_image, tf.uint8)
        generated_image = tf.image.encode_jpeg(generated_image)

        tf.io.write_file("./output/output.jpg", generated_image)

def save_image(img, path):
    # clip image between 0, 1
    img = np.clip(img, 0, 1)
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

class SRGAN:
    def __init__(self, model_gen_path, lr_dim=64, hr_dim=256):
        self.model_gen_path = model_gen_path
        self.model = self.load_model()
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim

    def load_model(self):
        model = keras.models.load_model(self.model_gen_path)
        return model

    def pack(self, img):
        pack_axis = np.argmin(img.shape[:-1])
        lower_dim = img.shape[pack_axis]
        dim_ratio = img.shape[0] / img.shape[1]
        # resize image to lower_dim, lower_dim
        img = img.numpy()
        img = img * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((int(lower_dim), int(lower_dim)))  
        img = np.array(img)
        img = img.astype(np.float32)

        return img, dim_ratio, pack_axis

    def unpack(self, img, dim_ratio, axis):
        # resize image to the higher dimension
        img = img * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        if axis == 1:
            img = img.resize((int(img.size[0] * dim_ratio), int(img.size[1])))

        if axis == 0:
            img = img.resize((int(img.size[0]), int(img.size[1] * dim_ratio)))
            
        return np.array(img, dtype=np.float32) / 255.0

    def downscale(self, img, dims):
        img = tf.image.resize(img, dims)
        return img

    def load_image_lr(self, path):
        im = Image.open(path)
        im_dims =  im.size

        # load the image in lr_dim, lr_dim
        im = im.resize((self.lr_dim, self.lr_dim))
        im = np.array(im).astype(np.float32)
        im = im[:, :, :3]
        return np.expand_dims((im / 255), axis=0)

    def upscale_64_256(self, image_path: str):
        image = self.load_image_lr(image_path)
        image = self.model(image)

        # image_file = Image.fromarray(ai_upscaled)
        # image_file.save("./output/test.jpg")

        save_image(image[0], "./output/test.jpg")