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
const voiceOutputBtn = document.getElementById('voiceOutputBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const voiceFeedbackToggle = document.getElementById('voiceFeedbackToggle');
const connectionStatus = document.getElementById('connectionStatus');
const statFps = document.getElementById('statFps');
const inferenceTimeEl = document.getElementById('inferenceTime');
const queueSizeEl = document.getElementById('queueSize');

// State Variables
let stream = null;
let isDetecting = false;
let ws = null;
let currentMode = 'standard';
let confidenceThreshold = 0.7;
let sentence = [];
let ctx = overlayCanvas.getContext('2d');
let frameInterval = null;
let voiceEnabled = true;
let lastSpokenGesture = '';
let gesturesList = [];

// WebSocket URL (Update with ngrok URL)
let WEBSOCKET_URL = 'ws://localhost:8765';

// Gesture mapping (will be updated from backend)
let gestureLabels = {};

// Initialize
async function init() {
    await loadCameras();
    setupEventListeners();
    animateOverlay();
    loadGesturesFromBackend();
    showToast('Welcome to SignSpeak Pro!', false);
}

// Load gestures from backend
async function loadGesturesFromBackend() {
    try {
        const response = await fetch('/models/metadata.json');
        if (response.ok) {
            const metadata = await response.json();
            gesturesList = metadata.actions || ['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please', 'sorry'];
            updateGesturesGrid();
        }
    } catch (error) {
        gesturesList = ['hello', 'thanks', 'iloveyou', 'yes', 'no', 'please', 'sorry', 'help', 'good', 'bad'];
        updateGesturesGrid();
    }
}

// Update gestures grid
function updateGesturesGrid() {
    const grid = document.getElementById('gesturesGrid');
    const gestureEmojis = {
        'hello': '👋', 'thanks': '👍', 'iloveyou': '🤟', 'yes': '✅', 'no': '❌',
        'please': '🙏', 'sorry': '😔', 'help': '🆘', 'eat': '🍔', 'drink': '🥤',
        'sleep': '😴', 'good': '👍', 'bad': '👎', 'happy': '😊', 'sad': '😢'
    };
    
    grid.innerHTML = gesturesList.slice(0, 12).map(gesture => `
        <div class="gesture-card">
            <div class="gesture-card-inner">
                <div class="gesture-card-front">
                    <div class="gesture-icon">${gestureEmojis[gesture] || '🤟'}</div>
                    <h3>${gesture.charAt(0).toUpperCase() + gesture.slice(1)}</h3>
                    <p>Sign language gesture</p>
                </div>
                <div class="gesture-card-back">
                    <i class="fas fa-hand-peace" style="font-size: 2rem;"></i>
                    <p style="margin-top: 10px;">Gesture: ${gesture}</p>
                </div>
            </div>
        </div>
    `).join('');
    
    document.getElementById('statGestures').textContent = gesturesList.length;
}

// Load cameras
async function loadCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        cameraSelect.innerHTML = '<option value="">Default Camera</option>';
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
    voiceOutputBtn.addEventListener('click', () => {
        voiceEnabled = !voiceEnabled;
        showToast(`Voice ${voiceEnabled ? 'ON' : 'OFF'}`, false);
        voiceOutputBtn.style.opacity = voiceEnabled ? '1' : '0.5';
    });
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    thresholdSlider.addEventListener('input', (e) => {
        confidenceThreshold = e.target.value / 100;
        thresholdValue.textContent = `${e.target.value}%`;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ threshold: confidenceThreshold }));
        }
    });
    
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            if (detectionModeSelect) detectionModeSelect.value = currentMode;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ mode: currentMode }));
            }
            showToast(`Switched to ${currentMode} mode`, false);
        });
    });
    
    detectionModeSelect?.addEventListener('change', (e) => {
        currentMode = e.target.value;
        modeBtns.forEach(btn => {
            if (btn.dataset.mode === currentMode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ mode: currentMode }));
        }
    });
    
    cameraSelect.addEventListener('change', async () => {
        if (isDetecting) await restartStream();
    });
    
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) settingsModal.classList.remove('active');
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            e.preventDefault();
            if (isDetecting) stopDetection();
            else startDetection();
        } else if (e.code === 'KeyS') {
            settingsModal.classList.add('active');
        }
    });
}

// Toggle theme
function toggleTheme() {
    const html = document.documentElement;
    if (html.getAttribute('data-theme') === 'light') {
        html.removeAttribute('data-theme');
        localStorage.setItem('theme', 'dark');
    } else {
        html.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
}

// Load saved theme
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') document.documentElement.setAttribute('data-theme', 'light');

// Start detection
async function startDetection() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined }
        });
        video.srcObject = stream;
        await video.play();
        
        updateConnectionStatus('connecting', 'Connecting...');
        
        ws = new WebSocket(WEBSOCKET_URL);
        
        ws.onopen = () => {
            console.log('✅ Connected to backend');
            updateConnectionStatus('connected', 'Connected');
            isDetecting = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
            ws.send(JSON.stringify({ mode: currentMode, threshold: confidenceThreshold }));
            sendFrames();
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateUI(data);
            if (data.fps) statFps.textContent = data.fps;
            if (data.inference_time) inferenceTimeEl.textContent = `${Math.round(data.inference_time)}ms`;
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus('error', 'Connection Error');
            showToast('Backend not reachable. Using demo mode.', true);
            startMockDetection();
        };
        
        ws.onclose = () => {
            console.log('WebSocket closed');
            updateConnectionStatus('disconnected', 'Disconnected');
            if (isDetecting) {
                startMockDetection();
            }
        };
        
    } catch (error) {
        console.error('Error starting detection:', error);
        showToast('Could not access camera. Please check permissions.', true);
    }
}

// Mock detection for demo
function startMockDetection() {
    isDetecting = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    startBtn.innerHTML = '<i class="fas fa-video"></i> Demo Mode';
    
    const mockGestures = ['hello', 'thanks', 'iloveyou', 'yes', 'no'];
    let idx = 0;
    
    if (frameInterval) clearInterval(frameInterval);
    frameInterval = setInterval(() => {
        if (!isDetecting) return;
        const gesture = mockGestures[idx % mockGestures.length];
        const scores = Array(gesturesList.length).fill(0.05);
        scores[idx % gesturesList.length] = 0.85 + Math.random() * 0.1;
        updateUI({
            prediction: gesture,
            confidence: 0.85,
            sentence: [gesture],
            confidence_scores: scores,
            mode: currentMode
        });
        idx++;
    }, 1500);
}

// Send frames
async function sendFrames() {
    async function sendFrame() {
        if (!isDetecting || !ws || ws.readyState !== WebSocket.OPEN) {
            if (isDetecting) requestAnimationFrame(sendFrame);
            return;
        }
        
        if (video.videoWidth > 0) {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.7);
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(imageData);
            }
        }
        
        requestAnimationFrame(sendFrame);
    }
    
    if (video.videoWidth > 0) {
        requestAnimationFrame(sendFrame);
    } else {
        video.addEventListener('loadedmetadata', () => requestAnimationFrame(sendFrame));
    }
}

// Update UI
function updateUI(data) {
    const { prediction, confidence, sentence: newSentence, confidence_scores } = data;
    
    if (newSentence && newSentence.length > 0) {
        sentence = newSentence;
        updateSentenceDisplay();
        if (sentence.length > 0 && voiceEnabled && sentence[sentence.length - 1] !== lastSpokenGesture) {
            speakText(`Detected ${sentence[sentence.length - 1]}`);
            lastSpokenGesture = sentence[sentence.length - 1];
            setTimeout(() => { lastSpokenGesture = ''; }, 2000);
        }
    }
    
    if (confidence_scores) {
        updateConfidenceBars(confidence_scores);
    }
    
    const badge = document.getElementById('predictionBadge');
    if (badge && prediction) {
        badge.innerHTML = `
            <span class="prediction-label">${prediction.charAt(0).toUpperCase() + prediction.slice(1)}</span>
            <span class="prediction-confidence">${Math.round(confidence * 100)}%</span>
        `;
    }
}

// Update sentence display
function updateSentenceDisplay() {
    if (sentence.length === 0) {
        sentenceDisplay.innerHTML = '<span class="placeholder">No gestures detected yet...</span>';
        return;
    }
    const sentenceText = sentence.map(g => g.charAt(0).toUpperCase() + g.slice(1)).join(' → ');
    sentenceDisplay.innerHTML = sentenceText;
    addToHistory(sentence[sentence.length - 1]);
}

// Update confidence bars
function updateConfidenceBars(scores) {
    const container = document.getElementById('confidenceBars');
    const topGestures = gesturesList.slice(0, 5);
    
    container.innerHTML = topGestures.map((gesture, idx) => {
        const score = scores[idx] || 0;
        return `
            <div class="confidence-item">
                <span class="label">${gesture.charAt(0).toUpperCase() + gesture.slice(1)}</span>
                <div class="bar-container">
                    <div class="bar" style="width: ${score * 100}%"></div>
                </div>
                <span class="value">${Math.round(score * 100)}%</span>
            </div>
        `;
    }).join('');
}

// Add to history
function addToHistory(gesture) {
    const historyList = document.getElementById('historyList');
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
        <span><i class="fas fa-hand-peace"></i> ${gesture.charAt(0).toUpperCase() + gesture.slice(1)}</span>
        <span>${new Date().toLocaleTimeString()}</span>
    `;
    historyList.insertBefore(historyItem, historyList.firstChild);
    while (historyList.children.length > 10) historyList.removeChild(historyList.lastChild);
    const placeholder = historyList.querySelector('.history-placeholder');
    if (placeholder) placeholder.remove();
}

// Clear history
function clearHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '<div class="history-placeholder">No predictions yet</div>';
    sentence = [];
    updateSentenceDisplay();
    showToast('History cleared!', false);
}

// Stop detection
function stopDetection() {
    isDetecting = false;
    
    if (frameInterval) clearInterval(frameInterval);
    if (ws) { ws.close(); ws = null; }
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    startBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
    updateConnectionStatus('disconnected', 'Disconnected');
    
    sentenceDisplay.innerHTML = '<span class="placeholder">No gestures detected yet...</span>';
    updateConfidenceBars(Array(gesturesList.length).fill(0));
    
    const badge = document.getElementById('predictionBadge');
    if (badge) badge.innerHTML = `<span class="prediction-label">Ready</span><span class="prediction-confidence">0%</span>`;
}

// Restart stream
async function restartStream() {
    if (isDetecting) {
        stopDetection();
        await startDetection();
    }
}

// Update connection status
function updateConnectionStatus(status, message) {
    connectionStatus.className = `connection-status ${status}`;
    connectionStatus.innerHTML = `<i class="fas fa-circle"></i> ${message}`;
}

// Speak text
function speakText(text) {
    if (!voiceEnabled) return;
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
    }
}

// Show toast
function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.borderLeftColor = isError ? '#ef4444' : '#10b981';
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// Animate overlay
function animateOverlay() {
    const draw = () => {
        if (!overlayCanvas || !video) return;
        overlayCanvas.width = video.clientWidth;
        overlayCanvas.height = video.clientHeight;
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        
        if (isDetecting) {
            const time = Date.now() / 1000;
            const scanY = (Math.sin(time * 2) + 1) / 2 * overlayCanvas.height;
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
            ctx.lineWidth = 2;
            ctx.moveTo(0, scanY);
            ctx.lineTo(overlayCanvas.width, scanY);
            ctx.stroke();
            
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.8)';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(10, 20);
            ctx.lineTo(10, 10);
            ctx.lineTo(20, 10);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(overlayCanvas.width - 10, 20);
            ctx.lineTo(overlayCanvas.width - 10, 10);
            ctx.lineTo(overlayCanvas.width - 20, 10);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(10, overlayCanvas.height - 20);
            ctx.lineTo(10, overlayCanvas.height - 10);
            ctx.lineTo(20, overlayCanvas.height - 10);
            ctx.stroke();
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

// Cursor glow
document.addEventListener('mousemove', (e) => {
    const glow = document.querySelector('.cursor-glow');
    if (glow) glow.style.transform = `translate(${e.clientX - 200}px, ${e.clientY - 200}px)`;
});

// Initialize
init();
