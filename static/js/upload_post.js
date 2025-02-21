document.getElementById("upload-form").addEventListener("submit", function(event) {
    let imageInput = document.getElementById("image-upload").files.length;
    let captionText = document.getElementById("caption").value.trim();
    let errorMessage = document.getElementById("error-message");

    if (!imageInput && !captionText) {
        errorMessage.style.display = "block";
        event.preventDefault(); // Stop form submission
    } else {
        errorMessage.style.display = "none";
    }
});

function previewImage(event) {
    let reader = new FileReader();
    reader.onload = function () {
        let imgPreview = document.getElementById("image-preview");
        imgPreview.src = reader.result;
        document.getElementById("image-preview-container").style.display = "flex";
    };
    reader.readAsDataURL(event.target.files[0]);
}

function removeImage() {
    document.getElementById("image-upload").value = "";
    document.getElementById("image-preview-container").style.display = "none";
}
