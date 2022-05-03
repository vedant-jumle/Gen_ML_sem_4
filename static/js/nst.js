const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;

if(output){
    document.getElementById('img-result').src="/static/NST_output.png";
    // left wala image = "NST_content.png"
    document.getElementById('img-upload1').src="/static/NST_content.png";
    document.getElementById('img-upload1').hidden=false;
    // right wala image = "NST_style.png"
    document.getElementById('img-upload').src="/static/NST_style.png";
    document.getElementById('img-upload').hidden=false;
    document.getElementsByClassName('upload-section')[0].style.padding="0%";
    document.getElementsByClassName('upload-section')[1].style.padding="0%";

    document.getElementById("upload-btn").hidden=true;
    document.getElementById("upload-btn1").hidden=true;
}

function loading(){
    document.getElementById('img-load').hidden=false;
    document.getElementById('img-result').hidden=true;
}

function loaded(){
    document.getElementById('img-result').hidden=false;
    document.getElementById('img-load').hidden=true;
}

function updateValue(){
    var value = document.getElementById('slider').value;
    document.getElementById('slider-value').innerHTML=value;
}