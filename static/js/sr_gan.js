const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;
console.log(output);

function readURL(event,mode){
    console.log(URL.createObjectURL(event.target.files[0]));
    if(mode===0){
        document.getElementById('img-upload').src = URL.createObjectURL(event.target.files[0]);
        document.getElementById('img-upload').hidden = false;
        document.getElementById('upload-btn').hidden = true;
        document.getElementsByClassName('upload-section')[0].style.padding = '0px';
        document.getElementsByClassName('upload-section')[0].style.border.radius = '0px';
    }else{
        document.getElementById('img-upload1').src = URL.createObjectURL(event.target.files[0]);
        document.getElementById('img-upload1').hidden = false;
        document.getElementById('upload-btn1').hidden = true;
        document.getElementsByClassName('upload-section')[1].style.padding = '0px';
        document.getElementsByClassName('upload-section')[1].style.border.radius = '0px';
    }
}

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
    document.getElementsByClassName('upload-section')[0].style.padding = '0px';
    document.getElementsByClassName('upload-section')[0].style.border.radius = '0px';
    document.getElementsByClassName('upload-section')[1].style.padding = '0px';
    document.getElementsByClassName('upload-section')[1].style.border.radius = '0px';
}