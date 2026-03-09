function runDetection(){

fetch("/detect")

.then(response => response.json())

.then(data => {

console.log("Server data:",data)

document.getElementById("time").innerText = data.time ?? "-"
document.getElementById("location").innerText = data.location ?? "-"
document.getElementById("helmet").innerText = data.helmet ?? "-"
document.getElementById("seatbelt").innerText = data.seatbelt ?? "-"
document.getElementById("activity").innerText = data.activity ?? "-"
document.getElementById("vehicle").innerText = data.vehicle ?? "-"
document.getElementById("violation").innerText = data.violation ?? "Safe"
document.getElementById("fine").innerText = data.fine ?? "0"

if(data.violation && data.violation !== "Safe"){

alert(
"⚠ TRAFFIC VIOLATION DETECTED\n\n"+
"Vehicle Number : "+data.vehicle+"\n"+
"Violation : "+data.violation+"\n"+
"Fine Amount : ₹"+data.fine
)

}

})

.catch(error=>{
console.log("Server error:",error)
})

}

setInterval(runDetection,9000)