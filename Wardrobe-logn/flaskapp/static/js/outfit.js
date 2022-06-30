function submit_this_form() {

}

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
            document.getElementById("btn-weather").style.display ="block";





        } else if (document.getElementById('no').checked) {
                document.getElementById("textDisplay4").style.display ="block";
                document.getElementById('event').style.display = "block";
                document.getElementById('travel').style.display = "block";
                document.getElementById('work').style.display = "block";
                document.getElementById('walk').style.display = "block";
                document.getElementById('event_name_1').style.display = "block";
                document.getElementById('event_name_2').style.display = "block";
                document.getElementById('event_name_3').style.display = "block";
                document.getElementById('event_name_4').style.display = "block";
                document.getElementById('btn-events').style.display = "block";




        }


}

function send_city(){

    document.getElementById("textDisplay4").style.display ="block";
    document.getElementById('event').style.display = "block";
    document.getElementById('walk').style.display = "block";
    document.getElementById('work').style.display = "block";
    document.getElementById('travel').style.display = "block";


    document.getElementById('event_name_1').style.display = "block";
    document.getElementById('event_name_2').style.display = "block";
    document.getElementById('event_name_3').style.display = "block";
    document.getElementById('event_name_4').style.display = "block";
    document.getElementById('btn-events').style.display = "block";


}


function checkButton() {

            document.getElementById('btn-show').style.display = "block";
            // document.getElementById('submit_btn').style.display = "none";

        }


