const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;

if(output!=undefined){
    // left wala image = "NST_content.png"
    document.getElementById('img-upload1').src="/static/NST_content.png";
    // right wala image = "NST_style.png"
    document.getElementById('img-upload').src="/static/NST_style.png";
}

function loading(){
    document.getElementById('img-result').src="/static/loading.gif";
}

function loaded(){
    document.getElementById('img-result').src="/static/NST_output.png";
}