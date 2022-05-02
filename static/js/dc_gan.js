const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;

if(output!=undefined){
    document.getElementById('output-image').src = "/static/DCGAN_output.png";
}