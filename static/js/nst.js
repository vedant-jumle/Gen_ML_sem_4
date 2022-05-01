const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;

if(output!=undefined){
    document.getElementById('img-result').src="static/img/NST_output.png";
}