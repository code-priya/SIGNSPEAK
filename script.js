// DOM Elements
const video = document.getElementById('videoFeed');
const overlayCanvas = document.getElementById('overlayCanvas');
const startBtn = document.getElementById('startDetection');
const stopBtn = document.getElementById('stopDetection');
const settingsBtn = document.getElementById('openSettings');
const settingsModal = document.getElementById('settingsModal');
const themeToggle = document.getElementById('themeToggle');
const sentenceDisplay = document.getElementById('sentenceDisplay');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const cameraSelect = document.getElementById('cameraSelect');
const detectionModeSelect = document.getElementById('detectionModeSelect');
const modeBtns = document.querySelectorAll('.mode-btn');
const confidenceValues = {
    hello: document.getElementById('confHello'),
    thanks: document.getElementById('confThanks'),
    iloveyou: document.getElementById('confIloveyou')
};
const confidenceBars = document.querySelectorAll('.bar');

// State Variables
let stream = null;
let isDetecting = false;
let ws = null;
let currentMode = 'standard';
let confidenceThreshold = 0.7;
let sentence = [];
let predictions = [];
let ctx = overlayCanvas.getContext('2d');

// IMPORTANT: Replace this with your ngrok URL after step 3
// You will update this URL after ngrok is running
let WEBSOCKET_URL = 'wss://YOUR_NGROK_URL.ngrok.io';  // ← UPDATE THIS

// Gesture mapping
const gestureMap = {
    0: 'hello',
    1: 'thanks',
    2: 'iloveyou'
};

const gestureLabels = {
    hello: 'Hello 👋',
    thanks: 'Thanks 👍',
    iloveyou: 'I Love You 🤟'
};

// Initialize
async function init() {
    await loadCameras();
    setupEventListeners();
    animateOverlay();
}

// Load available cameras
async function loadCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading cameras:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    startBtn.addEventListener('click', startDetection);
    stopBtn.addEventListener('click', stopDetection);
    settingsBtn.addEventListener('click', () => settingsModal.classList.add('active'));
    document.querySelector('.modal-close').addEventListener('click', () => settingsModal.classList.remove('active'));
    themeToggle.addEventListener('click', toggleTheme);
    
    thresholdSlider.addEventListener('input', (e) => {
        confidenceThreshold = e.target.value / 100;
        thresholdValue.textContent = `${e.target.value}%`;
    });
    
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            if (detectionModeSelect) {
                detectionModeSelect.value = currentMode;
            }
        });
    });
    
    detectionModeSelect?.addEventListener('change', (e) => {
        currentMode = e.target.value;
        modeBtns.forEach(btn => {
            if (btn.dataset.mode === currentMode) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    });
    
    cameraSelect.addEventListener('change', async () => {
        if (isDetecting) {
            await restartStream();
        }
    });
    
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.remove('active');
        }
    });
}

// Toggle theme
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    if (currentTheme === 'light') {
        html.removeAttribute('data-theme');
        localStorage.setItem('theme', 'dark');
    } else {
        html.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
}

// Load saved theme
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
}

// Start detection
async function startDetection() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined }
        });
        video.srcObject = stream;
        
        await video.play();
        
        // Connect WebSocket using ngrok URL
        ws = new WebSocket(WEBSOCKET_URL);
        
        ws.onopen = () => {
            console.log('WebSocket connected to ngrok');
            isDetecting = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateUI(data);
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            showError('Connection error. Make sure the backend is running and ngrok URL is correct.');
        };
        
        ws.onclose = () => {
            console.log('WebSocket closed');
            if (isDetecting) {
                showError('Connection lost. Please refresh and restart detection.');
            }
        };
        
        // Send video frames
        sendFrames();
        
    } catch (error) {
        console.error('Error starting detection:', error);
        showError('Could not access camera. Please check permissions.');
    }
}

// Send frames to backend via WebSocket
async function sendFrames() {
    async function sendFrame() {
        if (!isDetecting || !ws || ws.readyState !== WebSocket.OPEN) {
            requestAnimationFrame(sendFrame);
            return;
        }
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert to base64 for sending
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(imageData);
        }
        
        requestAnimationFrame(sendFrame);
    }
    
    // Wait for video to have dimensions
    if (video.videoWidth > 0) {
        requestAnimationFrame(sendFrame);
    } else {
        video.addEventListener('loadedmetadata', () => requestAnimationFrame(sendFrame));
    }
}

// Update UI with predictions
function updateUI(data) {
    const { prediction, confidence, sentence: newSentence, confidence_scores } = data;
    
    // Update sentence display
    if (newSentence && newSentence.length > 0) {
        sentence = newSentence;
        updateSentenceDisplay();
    }
    
    // Update confidence bars
    if (confidence_scores) {
        updateConfidenceBars(confidence_scores);
    }
    
    // Update prediction badge
    const badge = document.getElementById('predictionBadge');
    if (badge && prediction) {
        badge.innerHTML = `
            <span class="prediction-label">${gestureLabels[prediction] || prediction}</span>
            <span class="prediction-confidence">${Math.round(confidence * 100)}%</span>
        `;
    }
    
    // Update confidence values
    if (confidenceValues[prediction]) {
        confidenceValues[prediction].textContent = `${Math.round(confidence * 100)}%`;
    }
}

// Update sentence display
function updateSentenceDisplay() {
    if (sentence.length === 0) {
        sentenceDisplay.innerHTML = '<span class="placeholder">No gestures detected yet...</span>';
        return;
    }
    
    const sentenceText = sentence.map(g => gestureLabels[g] || g).join(' → ');
    sentenceDisplay.innerHTML = sentenceText;
    
    // Add to history
    addToHistory(sentence[sentence.length - 1]);
}

// Update confidence bars
function updateConfidenceBars(scores) {
    const gestures = ['hello', 'thanks', 'iloveyou'];
    gestures.forEach((gesture, index) => {
        const score = scores[index] || 0;
        const bar = document.querySelector(`.bar[data-gesture="${gesture}"]`);
        const valueSpan = document.getElementById(`conf${gesture.charAt(0).toUpperCase() + gesture.slice(1)}`);
        
        if (bar) {
            bar.style.width = `${score * 100}%`;
        }
        if (valueSpan) {
            valueSpan.textContent = `${Math.round(score * 100)}%`;
        }
    });
}

// Add to history
function addToHistory(gesture) {
    const historyList = document.getElementById('historyList');
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
        <span>${gestureLabels[gesture] || gesture}</span>
        <span>${new Date().toLocaleTimeString()}</span>
    `;
    
    historyList.insertBefore(historyItem, historyList.firstChild);
    
    // Keep only last 10 items
    while (historyList.children.length > 10) {
        historyList.removeChild(historyList.lastChild);
    }
    
    // Remove placeholder if exists
    const placeholder = historyList.querySelector('.history-placeholder');
    if (placeholder) {
        placeholder.remove();
    }
}

// Stop detection
function stopDetection() {
    isDetecting = false;
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
    
    // Clear UI
    sentenceDisplay.innerHTML = '<span class="placeholder">No gestures detected yet...</span>';
    updateConfidenceBars([0, 0, 0]);
    
    const badge = document.getElementById('predictionBadge');
    if (badge) {
        badge.innerHTML = `
            <span class="prediction-label">Detecting...</span>
            <span class="prediction-confidence" id="confidenceValue">0%</span>
        `;
    }
}

// Restart stream
async function restartStream() {
    if (isDetecting) {
        stopDetection();
        await startDetection();
    }
}

// Animate overlay
function animateOverlay() {
    if (!ctx) return;
    
    const draw = () => {
        if (!overlayCanvas || !video) return;
        
        overlayCanvas.width = video.clientWidth;
        overlayCanvas.height = video.clientHeight;
        
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (isDetecting) {
            // Draw scanning effect
            const time = Date.now() / 1000;
            const scanY = (Math.sin(time * 2) + 1) / 2 * overlayCanvas.height;
            
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
            ctx.lineWidth = 2;
            ctx.moveTo(0, scanY);
            ctx.lineTo(overlayCanvas.width, scanY);
            ctx.stroke();
            
            // Draw corner brackets
            const cornerSize = 30;
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.8)';
            ctx.lineWidth = 3;
            
            // Top-left
            ctx.beginPath();
            ctx.moveTo(10, 20);
            ctx.lineTo(10, 10);
            ctx.lineTo(20, 10);
            ctx.stroke();
            
            // Top-right
            ctx.beginPath();
            ctx.moveTo(overlayCanvas.width - 10, 20);
            ctx.lineTo(overlayCanvas.width - 10, 10);
            ctx.lineTo(overlayCanvas.width - 20, 10);
            ctx.stroke();
            
            // Bottom-left
            ctx.beginPath();
            ctx.moveTo(10, overlayCanvas.height - 20);
            ctx.lineTo(10, overlayCanvas.height - 10);
            ctx.lineTo(20, overlayCanvas.height - 10);
            ctx.stroke();
            
            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(overlayCanvas.width - 10, overlayCanvas.height - 20);
            ctx.lineTo(overlayCanvas.width - 10, overlayCanvas.height - 10);
            ctx.lineTo(overlayCanvas.width - 20, overlayCanvas.height - 10);
            ctx.stroke();
        }
        
        requestAnimationFrame(draw);
    };
    
    draw();
}

// Show error message
function showError(message) {
    const badge = document.getElementById('predictionBadge');
    if (badge) {
        badge.innerHTML = `
            <span class="prediction-label" style="color: #ef4444;">⚠️ Error</span>
            <span class="prediction-confidence" style="font-size: 10px;">${message}</span>
        `;
        setTimeout(() => {
            if (!isDetecting) {
                badge.innerHTML = `
                    <span class="prediction-label">Detecting...</span>
                    <span class="prediction-confidence" id="confidenceValue">0%</span>
                `;
            }
        }, 3000);
    }
}

// Cursor glow effect
document.addEventListener('mousemove', (e) => {
    const glow = document.querySelector('.cursor-glow');
    if (glow) {
        glow.style.transform = `translate(${e.clientX - 200}px, ${e.clientY - 200}px)`;
    }
});

// Initialize
init();// DOM Elements

