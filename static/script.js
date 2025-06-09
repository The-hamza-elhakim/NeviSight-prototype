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
        const file = fileInput.files[0];
        console.log("📁 File selected:", file);

        if (!file) {
            errorMsg.style.display = 'none';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
            return;
        }

        const ext = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['jpg', 'jpeg', 'png', 'dcm'];
        console.log("📦 File extension:", ext);

        // Step 1: Extension validation
        if (!allowedExtensions.includes(ext)) {
            errorMsg.style.display = 'block';
            errorMsg.innerText = 'File type not supported.';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
            return;
        }

        // Step 2: Send file to backend for fundus validation
        const formData = new FormData();
        formData.append('file', file);

        fetch('/validate-image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(result => {
            if (result.status === 'ok') {
                errorMsg.style.display = 'none';
                submitBtn.disabled = false;
                metadataFields.style.display = ext === 'dcm' ? 'none' : 'block';
                console.log("✅ Fundus image validated successfully.");
            } else {
                errorMsg.style.display = 'block';
                errorMsg.innerText = result.message || 'This is not a valid fundus image.';
                submitBtn.disabled = true;
                metadataFields.style.display = 'none';
                fileInput.value = ''; // clear invalid file
                console.warn("⚠️ Fundus image rejected by backend.");
            }
        })
        .catch(err => {
            console.error("❌ Error during validation:", err);
            errorMsg.style.display = 'block';
            errorMsg.innerText = 'Error validating image.';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
            fileInput.value = '';
        });
    });
});
