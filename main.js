const startCamBtn = document.getElementById('startCam');
const stopCamBtn = document.getElementById('stopCam');
const captureBtn = document.getElementById('capture');
const sendBtn = document.getElementById('send');
const fileInput = document.getElementById('fileInput');
const video = document.getElementById('video');
const preview = document.getElementById('preview');
const predSpan = document.getElementById('pred');

let stream = null;

// Set canvas size to match model input
preview.width = 128;
preview.height = 128;

// Start the camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        video.srcObject = stream;
        await video.play();
    } catch (e) {
        alert('Could not start camera. Check permissions.');
        console.error(e);
    }
}

// Stop the camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        video.srcObject = null;
        stream = null;
    }
}

// Capture current video frame to preview canvas
function captureToPreview() {
    const ctx = preview.getContext('2d');
    const w = preview.width, h = preview.height;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, w, h);

    if (video.videoWidth) {
        const ar = video.videoWidth / video.videoHeight;
        let dw = w, dh = h;
        if (ar > 1) dh = Math.round(w / ar);
        else dw = Math.round(h * ar);

        ctx.drawImage(video, (w - dw) / 2, (h - dh) / 2, dw, dh);
    }
}

// Handle file upload and draw it on preview canvas
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
        const ctx = preview.getContext('2d');
        const w = preview.width, h = preview.height;

        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, w, h);

        const ar = img.width / img.height;
        let dw = w, dh = h;
        if (ar > 1) dh = Math.round(w / ar);
        else dw = Math.round(h * ar);

        ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
    };
    img.src = URL.createObjectURL(file);
});

// Send the preview canvas to the server for prediction
async function sendPreview() {
    const dataUrl = preview.toDataURL('image/png');
    predSpan.textContent = '...';

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        });

        const j = await res.json();
        if (res.ok) {
            predSpan.textContent = j.prediction + (j.score ? ` (score: ${j.score.toFixed(3)})` : '');
        } else {
            predSpan.textContent = 'Error: ' + (j.error || 'unknown');
        }
    } catch (e) {
        predSpan.textContent = 'Network error';
        console.error(e);
    }
}

// Button events
startCamBtn.onclick = startCamera;
stopCamBtn.onclick = stopCamera;
captureBtn.onclick = captureToPreview;
sendBtn.onclick = sendPreview;

// Initialize blank preview
(function () {
    const ctx = preview.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, preview.width, preview.height);
})();

