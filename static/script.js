document.addEventListener('DOMContentLoaded', function () {
    console.log("🧠 script.js is loaded and running.");

    const modelSelect = document.querySelector('select[name="model"]');

    // Preload the default model (e.g., "unet")
    fetch('/preload-model/unet')
        .then(res => res.json())
        .then(data => console.log("✅ Default model preloaded:", data.message))
        .catch(err => console.error("❌ Error preloading default model:", err));

    // Dynamically load model list from backend
    fetch('/get-models')
        .then(res => res.json())
        .then(models => {
            if (models.length === 0) {
                const option = document.createElement('option');
                option.text = 'No models available';
                option.disabled = true;
                modelSelect.appendChild(option);
                return;
            }

            modelSelect.innerHTML = ''; // Clear existing options

            models.forEach((model, index) => {
                const option = document.createElement('option');
                option.value = model.id;
                option.text = model.name;
                if (index === 0) option.selected = true; // Auto-select first model
                modelSelect.appendChild(option);
            });

            modelSelect.addEventListener('change', function () {
                const selectedModel = modelSelect.value;
                fetch(`/preload-model/${selectedModel}`)
                    .then(res => res.json())
                    .then(data => console.log(`✅ Preloaded ${selectedModel}:`, data.message))
                    .catch(err => console.error(`❌ Failed to preload ${selectedModel}:`, err));
            });
        })
        .catch(err => {
            console.error("❌ Failed to load models:", err);
            const option = document.createElement('option');
            option.text = 'Error loading models';
            option.disabled = true;
            modelSelect.appendChild(option);
        });

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
    });

});
