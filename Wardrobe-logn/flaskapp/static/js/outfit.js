
function myFunction(){
    document.getElementById("piece1").style.display ="block";
    document.getElementById("piece2").style.display ="block";
    document.getElementById("piece3").style.display ="block";
    document.getElementById("outfit1").style.display ="block";
    document.getElementById("outfit2").style.display ="block";
    document.getElementById("outfit3").style.display ="block";
    document.getElementById("submit").style.display ="block";
    document.getElementById("textDisplay1").style.display ="block";
    document.getElementById("textDisplay2").style.display ="block";
    document.getElementById('btn-show').style.display = "none";
}

function send(){

        if (document.getElementById('yes').checked) {
            document.getElementById("textDisplay3").style.display ="block";
            document.getElementById("city_1").style.display ="block";
            document.getElementById("city_2").style.display ="block";
            document.getElementById("city_3").style.display ="block";
            document.getElementById("city_name_1").style.display ="block";
            document.getElementById("city_name_2").style.display ="block";
            document.getElementById("city_name_3").style.display ="block";
            document.getElementById('btn-ok').style.display = "block";
            document.getElementById('btn-next').style.display = "none";

        } else if (document.getElementById('no').checked) {
                document.getElementById("textDisplay4").style.display ="block";
                document.getElementById('event').style.display = "block";
                document.getElementById('party').style.display = "block";
                document.getElementById('work').style.display = "block";
                document.getElementById('home').style.display = "block";
                document.getElementById('casual').style.display = "block";
                document.getElementById('event_name_1').style.display = "block";
                document.getElementById('event_name_2').style.display = "block";
                document.getElementById('event_name_3').style.display = "block";
                document.getElementById('event_name_4').style.display = "block";
                document.getElementById('event_name_5').style.display = "block";
                document.getElementById(' submit_btn').style.display = "block";
        }


}

function send_city(){

    document.getElementById("textDisplay4").style.display ="block";
    document.getElementById('btn-ok').style.display = "none";
    document.getElementById('event').style.display = "block";
    document.getElementById('party').style.display = "block";
    document.getElementById('work').style.display = "block";
    document.getElementById('home').style.display = "block";
    document.getElementById('casual').style.display = "block";
    document.getElementById('submit_btn').style.display = "block";
    document.getElementById('event_name_1').style.display = "block";
    document.getElementById('event_name_2').style.display = "block";
    document.getElementById('event_name_3').style.display = "block";
    document.getElementById('event_name_4').style.display = "block";
    document.getElementById('event_name_5').style.display = "block";



}


   function checkButton() {
            if(document.getElementById('event').checked) {
                document.getElementById("disp").innerHTML
                    = document.getElementById("event").value
                    + " radio button is checked";
            }
            else if(document.getElementById('casual').checked) {
                document.getElementById("disp").innerHTML
                    = document.getElementById("casual").value
                    + " radio button is checked";
            }
            else if(document.getElementById('work').checked) {
                document.getElementById("disp").innerHTML
                    = document.getElementById("work").value
                    + " radio button is checked";
            }
            else if(document.getElementById('party').checked) {
                document.getElementById("disp").innerHTML
                    = document.getElementById("party").value
                    + " radio button is checked";
            }
              else if(document.getElementById('home').checked) {
                document.getElementById("disp").innerHTML
                    = document.getElementById("home").value
                    + " radio button is checked";
            }
            else {
                document.getElementById("error").innerHTML
                    = "You have not selected any season";
            }
            document.getElementById('btn-show').style.display = "block";
        }