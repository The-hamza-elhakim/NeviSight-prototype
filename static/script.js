document.addEventListener('DOMContentLoaded', function () {
    console.log("🧠 script.js is loaded and running.");

    const modelSelect = document.querySelector('select[name="model"]');
    const fileInput = document.getElementById('fileInput');
    const errorMsg = document.getElementById('error-message');
    const submitBtn = document.getElementById('submitBtn');
    const metadataFields = document.getElementById('metadataFields');
    const form = document.querySelector('form');
    const loadingMessage = document.getElementById('loadingMessage');

    submitBtn.disabled = true;

    // Load models
    fetch('/get-models')
        .then(res => res.json())
        .then(models => {
            modelSelect.innerHTML = '';
            models.forEach((model, index) => {
                const option = document.createElement('option');
                option.value = model.id;
                option.text = model.name;
                if (index === 0) option.selected = true;
                modelSelect.appendChild(option);
            });
        });

    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];
        if (!file) {
            errorMsg.style.display = 'none';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
            return;
        }

        const ext = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['jpg', 'jpeg', 'png', 'dcm'];

        if (!allowedExtensions.includes(ext)) {
            errorMsg.style.display = 'block';
            errorMsg.innerText = 'File type not supported.';
            submitBtn.disabled = true;
            metadataFields.style.display = 'none';
            return;
        }

        errorMsg.style.display = 'none';
        submitBtn.disabled = false;
        metadataFields.style.display = ext === 'dcm' ? 'none' : 'block';

        // Dynamically require MRN only for non-DICOM
        const mrnInput = document.querySelector('input[name="mrn"]');
        mrnInput.required = ext !== 'dcm';
    });

    form.addEventListener('submit', function () {
        loadingMessage.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.innerText = 'Uploading...';
    });
});
