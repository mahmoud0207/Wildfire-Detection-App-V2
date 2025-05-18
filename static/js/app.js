document.addEventListener('DOMContentLoaded', () => {
    let currentImage = null;
    let currentModel = null;
    let analysisResult = null;
    let objectURL = null;

    const elements = {
        imageInput: document.getElementById('imageInput'),
        modelInput: document.getElementById('modelInput'),
        modelUrl: document.getElementById('modelUrl'),
        runButton: document.getElementById('runButton'),
        downloadButton: document.getElementById('downloadButton'),
        imagePreview: document.getElementById('imagePreview'),
        resultDisplay: document.getElementById('resultDisplay'),
        resultContainer: document.getElementById('resultContainer'),
        status: document.getElementById('statusMessages'),
        progressBar: document.getElementById('progressBar'),
        confidenceDisplay: document.getElementById('confidenceDisplay'),
        resultText: document.getElementById('resultText')
    };

    // Event Listeners
    document.getElementById('uploadImageBtn').addEventListener('click', handleImageUpload);
    document.getElementById('uploadModelBtn').addEventListener('click', handleModelUpload);
    document.getElementById('loadUrlBtn').addEventListener('click', handleModelURL);
    elements.runButton.addEventListener('click', runAnalysis);
    elements.downloadButton.addEventListener('click', handleDownload);
    elements.imageInput.addEventListener('change', previewImage);

    async function handleImageUpload() {
        if (!elements.imageInput.files.length) return showError('Select an image first');
        disableUI(true);
        
        try {
            showProgress('Uploading image...', 20);
            const formData = new FormData();
            formData.append('file', elements.imageInput.files[0]);

            const response = await fetch('/upload-image', {
                method: 'POST',
                body: formData
            });

            const data = await handleResponse(response);
            currentImage = data.filepath;
            
            // If server generated a preview, use it
            if (data.preview_path) {
                updatePreview(`/results/${data.preview_path}`);
            }
            
            checkReadyState();
            showSuccess('Image uploaded successfully');
        } catch (error) {
            showError(error.message);
        } finally {
            disableUI(false);
        }
    }

    async function handleModelUpload() {
        if (!elements.modelInput.files.length) return showError('Select a model first');
        disableUI(true);
        
        try {
            showProgress('Uploading model...', 30);
            const formData = new FormData();
            formData.append('file', elements.modelInput.files[0]);

            const response = await fetch('/upload-model', {
                method: 'POST',
                body: formData
            });

            const data = await handleResponse(response);
            currentModel = data.model_path;
            checkReadyState();
            showSuccess('Model uploaded successfully');
        } catch (error) {
            showError(error.message);
        } finally {
            disableUI(false);
        }
    }

    async function runAnalysis() {
        if (!currentImage || !currentModel) return showError('Upload both files first');
        disableUI(true);
        
        try {
            showProgress('Analyzing image for wildfire detection...', 50);
            const response = await fetch('/run-model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_path: currentImage,
                    model_path: currentModel
                })
            });

            const data = await handleResponse(response);
            analysisResult = data;
            
            // Handle undefined values
            const prediction = data.prediction || 'unknown';
            const confidence = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '0%';
            
            // Display result directly on the page
            displayResult(prediction, confidence);
            
            // Update the preview with the heatmap image
            if (data.heatmap_url) {
                updatePreview(data.heatmap_url);
                elements.downloadButton.disabled = false;
            }
            
            showSuccess(`Analysis complete: ${formatPrediction(prediction)} (${confidence})`);
        } catch (error) {
            showError(error.message);
        } finally {
            disableUI(false);
        }
    }

    function displayResult(prediction, confidence) {
    elements.resultContainer.style.display = 'block';
    const confidenceContainer = document.getElementById('confidenceContainer');
    
    // Set the result text
    elements.resultText.textContent = formatPrediction(prediction);
    
    // Set appropriate classes based on the prediction
    if (prediction === 'wildfire') {
        elements.resultText.className = 'text-danger fw-bold';
        elements.resultContainer.className = 'alert alert-danger';
        confidenceContainer.style.display = 'block';
        elements.confidenceDisplay.textContent = confidence;
    } else if (prediction === 'no_wildfire') {
        elements.resultText.className = 'text-success fw-bold';
        elements.resultContainer.className = 'alert alert-success';
        confidenceContainer.style.display = 'none';
    } else {
        elements.resultText.className = 'text-warning fw-bold';
        elements.resultContainer.className = 'alert alert-warning';
        confidenceContainer.style.display = 'none';
    }
}

    function previewImage() {
        if (!elements.imageInput.files.length) return;
        
        const file = elements.imageInput.files[0];
        
        // Clear previous preview
        if (objectURL) {
            URL.revokeObjectURL(objectURL);
            objectURL = null;
        }
        
        elements.imagePreview.style.display = 'none';
        
        // Create a new object URL for immediate preview
        objectURL = URL.createObjectURL(file);
        
        // Set up onload to prevent rendering issues
        elements.imagePreview.onload = () => {
            elements.imagePreview.style.display = 'block';
        };
        
        // Set the source to the object URL
        elements.imagePreview.src = objectURL;
        
        // Add file info to status
        elements.status.innerHTML = `<div class="alert alert-info">Ready to upload: ${file.name} (${formatFileSize(file.size)})</div>`;
        
        // Reset result display
        elements.resultContainer.style.display = 'none';
    }

    // Helper functions
    function handleResponse(response) {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'Server error');
            });
        }
        return response.json();
    }

    function checkReadyState() {
        elements.runButton.disabled = !(currentImage && currentModel);
    }

    function disableUI(state) {
        [
            document.getElementById('uploadImageBtn'),
            document.getElementById('uploadModelBtn'),
            document.getElementById('loadUrlBtn'),
            elements.runButton,
            elements.downloadButton
        ].forEach(btn => {
            if (btn) btn.disabled = state;
        });
    }

    function showProgress(message, percent) {
        elements.progressBar.style.display = 'block';
        const progressBarElement = elements.progressBar.querySelector('.progress-bar');
        if (progressBarElement) {
            progressBarElement.style.width = `${percent}%`;
            progressBarElement.setAttribute('aria-valuenow', percent);
        }
        elements.status.innerHTML = `<div class="alert alert-info">${message}</div>`;
    }

    function showSuccess(message) {
        elements.progressBar.style.display = 'none';
        elements.status.innerHTML = `<div class="alert alert-success">${message}</div>`;
    }

    function showError(message) {
        elements.progressBar.style.display = 'none';
        elements.status.innerHTML = `<div class="alert alert-danger">${message}</div>`;
    }

    function updatePreview(url) {
        if (!url) return;
        
        // Always add a cache-busting parameter to avoid browser caching
        const cacheBuster = `?t=${Date.now()}`;
        elements.imagePreview.src = url + cacheBuster;
        
        // Set up onload handler to ensure we only show the image when it's ready
        elements.imagePreview.onload = () => {
            elements.imagePreview.style.display = 'block';
        };
        
        // In case of load error
        elements.imagePreview.onerror = () => {
            elements.imagePreview.style.display = 'none';
            showError('Failed to load preview image');
        };
    }

    function handleDownload() {
    if (!analysisResult?.text_result) return;
    
    // Create text content
    const textContent = `Wildfire Detection Report
=============================
Prediction: ${analysisResult.text_result.prediction}
Confidence: ${analysisResult.text_result.confidence}
Model Used: ${analysisResult.text_result.model_used}
Image Dimensions: ${analysisResult.text_result.image_dimensions.width}x${analysisResult.text_result.image_dimensions.height}
Analysis Timestamp: ${analysisResult.text_result.timestamp}

Conclusion: ${analysisResult.text_result.prediction} with ${analysisResult.text_result.confidence} confidence`;

    // Create download
    const blob = new Blob([textContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const filename = `wildfire_report_${Date.now()}.txt`;
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showSuccess(`Downloaded report as "${filename}"`);
}

    function formatPrediction(pred) {
        return (pred || 'unknown')
            .replace(/(?:^|\s)\S/g, a => a.toUpperCase())
            .replace('_', ' ');
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async function handleModelURL() {
        const url = elements.modelUrl.value.trim();
        if (!url) return showError('Enter valid URL');
        disableUI(true);
        
        try {
            showProgress('Loading model from URL...', 30);
            const response = await fetch('/upload-model-url', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ url })
            });

            const data = await handleResponse(response);
            
            if (data.error && !data.model_path) {
                showError(data.error);
                if (data.message) {
                    elements.status.innerHTML += `<div class="alert alert-info mt-2">${data.message}</div>`;
                }
                return;
            }
            
            currentModel = data.model_path;
            checkReadyState();
            showSuccess('Model loaded from URL');
        } catch (error) {
            showError(error.message);
        } finally {
            disableUI(false);
        }
    }
});