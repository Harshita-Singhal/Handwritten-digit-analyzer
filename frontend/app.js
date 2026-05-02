const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const predictionDisplay = document.getElementById('predictionDisplay');
const confidenceText = document.getElementById('confidenceText');
const confidenceFill = document.getElementById('confidenceFill');

let isDrawing = false;

// Setup Canvas
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

clearBtn.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionDisplay.innerHTML = '<span class="placeholder">Awaiting input...</span>';
    confidenceText.innerText = "0%";
    confidenceFill.style.width = "0%";
    confidenceFill.style.backgroundColor = "var(--accent)";
});

analyzeBtn.addEventListener('click', async () => {
    predictionDisplay.innerHTML = '<span class="placeholder">Analyzing...</span>';
    confidenceText.innerText = "Processing...";
    
    const imageData = canvas.toDataURL('image/png');

    try {
        // 👇 YAHAN APNA RENDER URL DAALEIN 👇
        const response = await fetch('https://AAPKA-RENDER-APP-NAAM.onrender.com/predict', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json' 
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.error) {
            console.error("Backend Error:", data.error);
            predictionDisplay.innerHTML = '<span class="placeholder" style="color: red;">Error</span>';
            return;
        }

        // Show Results
        predictionDisplay.innerHTML = data.prediction;
        confidenceText.innerText = `${data.confidence}%`;
        confidenceFill.style.width = `${data.confidence}%`;

        // Change progress bar color based on accuracy
        if (data.confidence > 90) {
            confidenceFill.style.backgroundColor = "#10b981"; // Green
        } else if (data.confidence > 60) {
            confidenceFill.style.backgroundColor = "#eab308"; // Yellow
        } else {
            confidenceFill.style.backgroundColor = "#ef4444"; // Red
        }

    } catch (error) {
        console.error("Connection Error:", error);
        predictionDisplay.innerHTML = '<span class="placeholder" style="color: #ef4444;">Connection Error</span>';
        confidenceText.innerText = "Error";
    }
});
