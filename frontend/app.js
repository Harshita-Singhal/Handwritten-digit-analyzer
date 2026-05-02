// UI Elements
const card = document.getElementById('flipCard');
const analyzeBtn = document.getElementById('analyzeBtn');
const flipBackBtn = document.getElementById('flipBackBtn');

const tabDraw = document.getElementById('tabDraw');
const tabUpload = document.getElementById('tabUpload');
const drawSection = document.getElementById('drawSection');
const uploadSection = document.getElementById('uploadSection');

const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
let myChart = null; // Graph variable

// --- TAB SWITCHING LOGIC ---
tabDraw.addEventListener('click', () => {
    tabDraw.classList.add('active'); tabUpload.classList.remove('active');
    drawSection.classList.add('active'); uploadSection.classList.remove('active');
});
tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active'); tabDraw.classList.remove('active');
    uploadSection.classList.add('active'); drawSection.classList.remove('active');
});

// --- IMAGE UPLOAD PREVIEW ---
fileInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

// --- CANVAS DRAWING LOGIC ---
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15; ctx.lineCap = "round"; ctx.strokeStyle = "black";

canvas.addEventListener('mousedown', (e) => { isDrawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); });
canvas.addEventListener('mousemove', (e) => { if (isDrawing) { ctx.lineTo(e.offsetX, e.offsetY); ctx.stroke(); } });
canvas.addEventListener('mouseup', () => isDrawing = false);
document.getElementById('clearBtn').addEventListener('click', () => { ctx.fillRect(0, 0, canvas.width, canvas.height); });


// --- REAL BACKEND CONNECTION & FLIP LOGIC ---

analyzeBtn.addEventListener('click', async () => {
    // 1. Get Selected Logic (Clean or Camera)
    const selectedLogic = document.querySelector('input[name="logic"]:checked').value;
    
    // 2. Check if we are sending Canvas Drawing OR Uploaded Image
    let imageData = "";
    if (drawSection.classList.contains('active')) {
        imageData = canvas.toDataURL('image/png');
    } else {
        if (!imagePreview.src || imagePreview.style.display === 'none') {
            alert("Please upload an image first!");
            return;
        }
        imageData = imagePreview.src;
    }

    analyzeBtn.innerText = "Processing...";

    try {
        // 👇 YAHAN APNA RENDER URL DAALEIN 👇
        const response = await fetch('https://handwritten-digit-analyzer.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: imageData, 
                logic: selectedLogic 
            })
        });

        const data = await response.json();

        if (data.error) {
            alert("Error from Backend: " + data.error);
            analyzeBtn.innerText = "🚀 Analyze Data";
            return;
        }

        // 3. Populate Extracted Digits on the back card
        const box = document.getElementById('extractedDigitsList');
        box.innerHTML = "";
        data.extracted_digits.forEach(num => {
            let span = document.createElement('span');
            span.innerText = num;
            box.appendChild(span);
        });

        // 4. Update Confidence Score
        document.getElementById('confidenceText').innerText = `${data.overall_confidence}%`;

        // 5. Build Graph with Frequencies
        buildGraph(data.frequencies);

        // 6. FLIP THE CARD!
        card.classList.add('is-flipped');
        analyzeBtn.innerText = "🚀 Analyze Data"; // Reset button text

    } catch (error) {
        console.error(error);
        alert("Backend se connect nahi ho paya. Please check your Render URL or wait for Render to wake up.");
        analyzeBtn.innerText = "🚀 Analyze Data";
    }
});

// Flip Back Button
flipBackBtn.addEventListener('click', () => {
    card.classList.remove('is-flipped');
    if(myChart) myChart.destroy(); // Clear old graph
});

// --- CHART.JS GRAPH FUNCTION ---
function buildGraph(dataArray) {
    const ctxChart = document.getElementById('resultsChart').getContext('2d');
    
    if(myChart) myChart.destroy(); // Destroy existing chart before drawing new one

    myChart = new Chart(ctxChart, {
        type: 'bar',
        data: {
            labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            datasets: [{
                label: 'Digit Count',
                data: dataArray,
                backgroundColor: 'rgba(59, 130, 246, 0.8)',
                borderColor: '#3b82f6',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
