from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from generator_models import *

ALLOWED_EXTENSIONS = set(['png'])
UPLOAD_FOLDER = 'static'

app = Flask(__name__, static_folder="static")
app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

print("Loading models...")
nst = StyleTransfer()
dcgan = DCGAN("./models/DCGAN/generator/generator_332")
srgan = SRGAN("./models/SRGAN/gen/generator-320")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/img/<filename>")
def img(filename):
    return redirect(url_for('static', filename=f"/{filename}"))


# NST
@app.route("/NST_page")
def NST_page():
    return render_template("style_transfer.html")

@app.route("/NST", methods=['POST'])
def NST():
    files = [request.files['NST_content'], request.files['NST_style']]
    img_paths = []
    for i, file in enumerate(files):
        filename = "NST_style.png" if i == 1 else "NST_content.png"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_paths.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # takes time
    nst.style_image(*img_paths)
    
    
    return redirect("/NST_page?output=true")


@app.route("/DCGAN_page")
def DCGAN_page():
    pass


@app.route("/DCGAN_generate")
def generate_dcgan():
    dcgan.generate()
    return redirect("/img/DCGAN_output.png")

@app.route("/SRGAN_page")
def SRGAN_page():
    return render_template("sr_gan.html")

@app.route("/SRGAN", methods=["POST"])
def generate_srgan():
    # file saving logic
    if request.method == 'POST':
        file = request.files['lr_image']
        filename = "SRGAN_input.png"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    srgan.upscale_64_256()
    return redirect("/SRGAN_page?output=true")


if __name__ == "__main__":
    app.run(debug=True)