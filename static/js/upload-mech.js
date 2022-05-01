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