const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;

if(output!=undefined){
    document.getElementById('output-img').src = "/static/DCGAN_output.png";
}