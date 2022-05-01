const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;
console.log(output);

if (output != undefined) {
    // if the filename is there, then remove the form and add a new dom element with lr and hr image
    document.getElementById('generate-btn').hidden = true;
    // input = "SRGAN_input.png", output = "SRGAN_output.png"
    document.getElementById('sr_gan_img').src = "/static/SRGAN_output.png";
    document.getElementById('sr_gan_img').hidden = false;
    document.getElementById('upload-btn').hidden = true;
    document.getElementById('actual-btn').hidden = true;
    document.getElementById('img-upload').src = "/static/SRGAN_input.png";
    document.getElementById('img-upload').hidden = false;
}