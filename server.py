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
    return redirect(url_for('static', filename='img/' + filename))


@app.route("/NST", methods=['POST'])
def NST():
    files = [request.files['NST_content'], request.files['NST_style']]
    img_paths = []
    for i, file in enumerate(files):
        filename = "NST_style.png" if i == 1 else "NST_content.png"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_paths.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    nst.style_image(*img_paths)
    
    return redirect()