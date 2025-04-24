$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview')
                    .css('background-image', 'url(' + e.target.result + ')')
                    .on('error', function() {
                        $(this).css('background-image', 'none')
                            .html('<div class="alert alert-warning">Failed to load image</div>');
                    });
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(data);
            },
            error: function() {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').html('<div class="alert alert-danger">Error processing image</div>');
            }
        });
    });

     $('#btn-color').click(function () {
        var form_data = new FormData($('#upload-file')[0]);


        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/color',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {


                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(data);

            },
        });
    });



});
