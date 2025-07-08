document.addEventListener('DOMContentLoaded', function () {

    const modelSelect = document.getElementById('modelSelect');
    const fileNameDisplay = document.getElementById('fileNameDisplay');

    // Fetch available models from backend
    fetch('/get-models')
        .then(response => response.json())
        .then(models => {
            modelSelect.innerHTML = ''; // Clear the "Loading..." option
            models.forEach((model, index) => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                if (index === 0) option.selected = true;
                modelSelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error("Error loading models:", error);
            modelSelect.innerHTML = '<option disabled>Error loading models</option>';
        });


    const fileInput = document.getElementById('fileInput');
    const errorContainer = document.getElementById('errorContainer');
    const uploadBtn = document.getElementById('uploadBtn');
    const metadataFields = document.getElementById('metadataFields');
    const mrnInput = document.getElementById('mrnInput');
    const dropZone = document.getElementById('dropZone');

    // Click on dropZone triggers file dialog
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });

    // Drag over styling
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    // Remove styling on drag leave
    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    // Handle dropped file
    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            // Trigger change event manually
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];
        if (!file) {
            errorContainer.style.display = 'none';
            uploadBtn.disabled = true;
            metadataFields.style.display = 'none';
            fileNameDisplay.style.display = 'none';
            return;
        }

        const ext = file.name.split('.').pop().toLowerCase();
        const allowedExtensions = ['jpg', 'jpeg', 'png', 'dcm'];

        if (!allowedExtensions.includes(ext)) {
            errorContainer.style.display = 'block';
            errorContainer.innerText = 'File type not supported.';
            uploadBtn.disabled = true;
            metadataFields.style.display = 'none';
            return;
        }

        fileNameDisplay.style.display = 'block';
        fileNameDisplay.innerText = `Selected File: ${file.name}`;


        errorContainer.style.display = 'none';
        uploadBtn.disabled = true;
        metadataFields.style.display = 'none';

        // Create FormData for AJAX request
        const formData = new FormData();
        formData.append('file', file);

        // Show spinner while validating
        uploadBtn.innerHTML = `Validating... <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;

        fetch('/validate-fundus', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(result => {
            if (result.valid) {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = 'Upload and Analyze';
                if (ext === 'dcm') {
                    metadataFields.style.display = 'none';
                    mrnInput.required = false;
                } else {
                    metadataFields.style.display = 'block';
                    mrnInput.required = true;
                }
                
            } else {
                errorContainer.style.display = 'block';
                errorContainer.innerText = 'Not a valid fundus image.';
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = 'Upload and Analyze';
                metadataFields.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Validation error:', error);
            errorContainer.style.display = 'block';
            errorContainer.innerText = 'Error validating image.';
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = 'Upload and Analyze';
            metadataFields.style.display = 'none';
        });
    });


    document.getElementById('uploadForm').addEventListener('submit', function () {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `Processing... <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;
    });
});
