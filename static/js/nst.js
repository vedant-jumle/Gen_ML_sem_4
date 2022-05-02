const output = Qs.parse(location.search, {
    ignoreQueryPrefix: true
}).output;

if(output!=undefined){
    document.getElementById('img-result').src="static/NST_output.png";
    // left wala image = "NST_content.png"
    // right wala image = "NST_style.png"
}