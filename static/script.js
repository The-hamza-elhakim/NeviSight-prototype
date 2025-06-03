document.addEventListener('DOMContentLoaded', function () {
    console.log("🧠 script.js is loaded and running.");

    const fileInput = document.getElementById('fileInput');
    const errorMsg = document.getElementById('error-message');
    const submitBtn = document.getElementById('submitBtn');
    const metadataFields = document.getElementById('metadataFields');

    submitBtn.disabled = true;

    if (!fileInput) {
        console.error("❌ fileInput not found in DOM");
        return;
    }

    fileInput.addEventListener('change', function () {
        console.log("📁 File selected:", fileInput.files[0]);

        const file = fileInput.files[0];
        if (!file) {
            errorMsg.style.display = 'none';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
            return;
        }

        const ext = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['jpg', 'jpeg', 'png', 'dcm'];

        console.log("📦 File extension:", ext);

        if (!allowedExtensions.includes(ext)) {
            errorMsg.style.display = 'block';
            errorMsg.innerText = 'File type not supported.';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
        } else {
            errorMsg.style.display = 'none';
            submitBtn.disabled = false;

            if (ext === 'dcm') {
                metadataFields.style.display = 'none';
            } else {
                metadataFields.style.display = 'block';
            }
        }
    });
});
