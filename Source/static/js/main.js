// Initialize all value displays
var i;
for (i = 0; i < 11q; i++) {
    document.getElementById("out" + i).innerHTML = document.getElementById("in" + i).value;
}

// Update the current slider value (each time you drag the slider handle)
function update(val,id) {
    document.getElementById(id).innerHTML = val;
}