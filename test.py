from generator_models import *
srgan = SRGAN("./models/SRGAN/gen/generator-320")
srgan.upscale_64_256("./images/real_small.jpeg")