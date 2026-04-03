// ==================== ULTIMATE SIGNSPEAK PRO ====================
// Complete JavaScript with all enhanced features

// DOM Elements
const video = document.getElementById('videoFeed');
const overlayCanvas = document.getElementById('overlayCanvas');
const startBtn = document.getElementById('startDetection');
const stopBtn = document.getElementById('stopDetection');
const settingsBtn = document.getElementById('openSettings');
const settingsModal = document.getElementById('settingsModal');
const statsModal = document.getElementById('statsModal');
const themeToggle = document.getElementById('themeToggle');
const dayNightToggle = document.getElementById('dayNightToggle');
const sentenceDisplay = document.getElementById('liveTranscription');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const cameraSelect = document.getElementById('cameraSelect');
const detectionModeSelect = document.getElementById('detectionModeSelect');
const modeBtns = document.querySelectorAll('.mode-btn');
const voiceCmdBtn = document.getElementById('voiceCmdBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const voiceFeedbackToggle = document.getElementById('voiceFeedbackToggle');
const connectionStatus = document.getElementById('connectionStatus');
const statFps = document.getElementById('statFps');
const statStreak = document.getElementById('statStreak');
const openStatsBtn = document.getElementById('openStats');
const fabMain = document.getElementById('fabMain');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

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
let wordsToday = 0;
let currentStreak = 0;
let bestStreak = 0;
let totalGestures = 0;
let sessionStartTime = Date.now();
let practiceSession = null;
let recognition = null;
let isListening = false;
let charts = {};

// ==================== 50+ GESTURES DATABASE ====================
const gesturesDB = [
    // Basics
    { name: 'Hello', emoji: '👋', description: 'Wave your hand', category: 'greetings', difficulty: 'easy' },
    { name: 'Goodbye', emoji: '👋', description: 'Wave goodbye', category: 'greetings', difficulty: 'easy' },
    { name: 'Thank You', emoji: '🙏', description: 'Hand to chin', category: 'polite', difficulty: 'easy' },
    { name: 'Please', emoji: '🤲', description: 'Circular motion on chest', category: 'polite', difficulty: 'medium' },
    { name: 'Sorry', emoji: '😔', description: 'Fist over heart', category: 'emotions', difficulty: 'medium' },
    { name: 'Yes', emoji: '👍', description: 'Nodding fist', category: 'affirmative', difficulty: 'easy' },
    { name: 'No', emoji: '👎', description: 'Thumb and fingers together', category: 'negative', difficulty: 'easy' },
    
    // Emotions
    { name: 'Love', emoji: '❤️', description: 'Cross arms over chest', category: 'emotions', difficulty: 'medium' },
    { name: 'Happy', emoji: '😊', description: 'Brush chest upward', category: 'emotions', difficulty: 'easy' },
    { name: 'Sad', emoji: '😢', description: 'Fingers down face', category: 'emotions', difficulty: 'medium' },
    { name: 'Angry', emoji: '😠', description: 'Claw hand motion', category: 'emotions', difficulty: 'hard' },
    { name: 'Excited', emoji: '🤩', description: 'Both hands shake', category: 'emotions', difficulty: 'medium' },
    { name: 'Tired', emoji: '😴', description: 'Hand on forehead', category: 'emotions', difficulty: 'easy' },
    
    // Actions
    { name: 'Eat', emoji: '🍽️', description: 'Bring hand to mouth', category: 'actions', difficulty: 'easy' },
    { name: 'Drink', emoji: '🥤', description: 'C-shape to mouth', category: 'actions', difficulty: 'easy' },
    { name: 'Sleep', emoji: '😴', description: 'Hand on cheek', category: 'actions', difficulty: 'easy' },
    { name: 'Work', emoji: '💼', description: 'Tap wrists together', category: 'actions', difficulty: 'medium' },
    { name: 'Study', emoji: '📚', description: 'Hand reading motion', category: 'actions', difficulty: 'medium' },
    { name: 'Play', emoji: '🎮', description: 'Thumbs up wiggle', category: 'actions', difficulty: 'easy' },
    { name: 'Help', emoji: '🆘', description: 'Thumb up on palm', category: 'emergency', difficulty: 'easy' },
    { name: 'Wait', emoji: '⏳', description: 'Hand wave stop', category: 'actions', difficulty: 'easy' },
    
    // People
    { name: 'Family', emoji: '👨‍👩‍👧', description: 'Circle with hands', category: 'people', difficulty: 'hard' },
    { name: 'Friend', emoji: '👫', description: 'Hook index fingers', category: 'people', difficulty: 'medium' },
    { name: 'Mother', emoji: '👩', description: 'Thumb to chin', category: 'people', difficulty: 'easy' },
    { name: 'Father', emoji: '👨', description: 'Thumb to forehead', category: 'people', difficulty: 'easy' },
    { name: 'Teacher', emoji: '👨‍🏫', description: 'Hand from forehead', category: 'people', difficulty: 'medium' },
    { name: 'Doctor', emoji: '👨‍⚕️', description: 'Tap wrist', category: 'people', difficulty: 'medium' },
    
    // Objects
    { name: 'Water', emoji: '💧', description: 'W hand to mouth', category: 'objects', difficulty: 'easy' },
    { name: 'Food', emoji: '🍕', description: 'Hand to mouth', category: 'objects', difficulty: 'easy' },
    { name: 'Money', emoji: '💰', description: 'Tap palm', category: 'objects', difficulty: 'medium' },
    { name: 'Car', emoji: '🚗', description: 'Steering wheel motion', category: 'objects', difficulty: 'medium' },
    { name: 'House', emoji: '🏠', description: 'Roof shape with hands', category: 'objects', difficulty: 'medium' },
    { name: 'Phone', emoji: '📱', description: 'C hand to ear', category: 'objects', difficulty: 'easy' },
    
    // Time
    { name: 'Today', emoji: '📅', description: 'Tap wrist', category: 'time', difficulty: 'easy' },
    { name: 'Tomorrow', emoji: '⏩', description: 'Thumb over shoulder', category: 'time', difficulty: 'medium' },
    { name: 'Yesterday', emoji: '⏪', description: 'Thumb back', category: 'time', difficulty: 'medium' },
    { name: 'Morning', emoji: '🌅', description: 'Arm across body', category: 'time', difficulty: 'hard' },
    { name: 'Night', emoji: '🌙', description: 'Hand over head', category: 'time', difficulty: 'medium' },
    
    // Advanced
    { name: 'Understand', emoji: '🧠', description: 'Point to head', category: 'advanced', difficulty: 'hard' },
    { name: 'Beautiful', emoji: '🌸', description: 'Open hand circle face', category: 'advanced', difficulty: 'hard' },
    { name: 'Promise', emoji: '🤝', description: 'X over heart', category: 'advanced', difficulty: 'hard' },
    { name: 'Freedom', emoji: '🕊️', description: 'Hands open wide', category: 'advanced', difficulty: 'medium' },
    { name: 'Peace', emoji: '☮️', description: 'Peace sign', category: 'advanced', difficulty: 'easy' },
    { name: 'Respect', emoji: '🙌', description: 'Hands together bow', category: 'advanced', difficulty: 'medium' },
    
    // Numbers 1-10
    { name: 'One', emoji: '1️⃣', description: 'Index finger up', category: 'numbers', difficulty: 'easy' },
    { name: 'Two', emoji: '2️⃣', description: 'Two fingers up', category: 'numbers', difficulty: 'easy' },
    { name: 'Three', emoji: '3️⃣', description: 'Three fingers up', category: 'numbers', difficulty: 'easy' },
    { name: 'Four', emoji: '4️⃣', description: 'Four fingers up', category: 'numbers', difficulty: 'easy' },
    { name: 'Five', emoji: '5️⃣', description: 'Open hand', category: 'numbers', difficulty: 'easy' },
    { name: 'Six', emoji: '6️⃣', description: 'Thumb and pinky', category: 'numbers', difficulty: 'easy' },
    { name: 'Seven', emoji: '7️⃣', description: 'Thumb, index, middle', category: 'numbers', difficulty: 'easy' },
    { name: 'Eight', emoji: '8️⃣', description: 'Thumb and index', category: 'numbers', difficulty: 'easy' },
    { name: 'Nine', emoji: '9️⃣', description: 'Curled index', category: 'numbers', difficulty: 'easy' },
    { name: 'Ten', emoji: '🔟', description: 'Thumbs up shake', category: 'numbers', difficulty: 'easy' }
];

// ==================== INITIALIZATION ====================
async function init() {
    await loadCameras();
    setupEventListeners();
    animateOverlay();
    updateFPS();
    loadStats();
    populateGesturesGrid();
    initVoiceCommands();
    initCharts();
    loadThemePreference();
    initPracticeMode();
    initFloatingMenu();
    showToast('Welcome to SignSpeak Ultimate! 🎉', false);
}

// Load saved stats
function loadStats() {
    const saved = localStorage.getItem('signspeak_ultimate');
    if (saved) {
        const stats = JSON.parse(saved);
        currentStreak = stats.streak || 0;
        bestStreak = stats.bestStreak || 0;
        totalGestures = stats.totalGestures || 0;
        statStreak.textContent = currentStreak;
    }
    updateStreak();
}

// Update streak
function updateStreak() {
    const lastActive = localStorage.getItem('lastActive');
    const today = new Date().toDateString();
    if (lastActive === today) return;
    
    if (lastActive && new Date(lastActive).getTime() === new Date(today).getTime() - 86400000) {
        currentStreak++;
    } else if (lastActive !== today) {
        currentStreak = 1;
    }
    
    if (currentStreak > bestStreak) bestStreak = currentStreak;
    statStreak.textContent = currentStreak;
    localStorage.setItem('lastActive', today);
    saveStats();
}

// Save stats
function saveStats() {
    const stats = {
        streak: currentStreak,
        bestStreak: bestStreak,
        totalGestures: totalGestures,
        lastUpdated: new Date().toISOString()
    };
    localStorage.setItem('signspeak_ultimate', JSON.stringify(stats));
}

// Initialize charts
function initCharts() {
    const ctx1 = document.getElementById('confidenceChart')?.getContext('2d');
    if (ctx1) {
        charts.confidence = new Chart(ctx1, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Confidence', data: [], borderColor: '#6366f1', tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }
}

// Populate gestures grid
function populateGesturesGrid(searchTerm = '', category = 'all') {
    const grid = document.getElementById('gesturesGrid');
    const filtered = gesturesDB.filter(g => 
        (category === 'all' || g.category === category) &&
        (g.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
         g.description.toLowerCase().includes(searchTerm.toLowerCase()))
    );
    
    grid.innerHTML = filtered.map(gesture => `
        <div class="gesture-card" data-gesture="${gesture.name.toLowerCase()}">
            <div class="gesture-icon">${gesture.emoji}</div>
            <h3>${gesture.name}</h3>
            <p>${gesture.description}</p>
            <small style="color: var(--text-secondary);">
                ${gesture.category} • ${gesture.difficulty}
            </small>
        </div>
    `).join('');
    
    document.getElementById('statGestures').textContent = gesturesDB.length;
}

// Initialize practice mode
function initPracticeMode() {
    const startPractice = document.getElementById('startPracticeBtn');
    if (startPractice) {
        startPractice.addEventListener('click', startPracticeSession);
    }
}

function startPracticeSession() {
    const mode = document.getElementById('practiceMode').value;
    const gestures = gesturesDB.filter(g => g.difficulty !== 'hard');
    let currentIndex = 0;
    let score = 0;
    let streak = 0;
    
    practiceSession = setInterval(() => {
        if (currentIndex >= gestures.length) {
            clearInterval(practiceSession);
            showToast(`Practice complete! Score: ${score}/${gestures.length}`, false);
            return;
        }
        
        const target = gestures[currentIndex];
        document.getElementById('targetGesture').innerHTML = `
            <span class="target-emoji">${target.emoji}</span>
            <span class="target-name">${target.name}</span>
        `;
        
        // Simulate practice feedback
        setTimeout(() => {
            const userScore = Math.random() > 0.3 ? 1 : 0;
            if (userScore === 1) {
                score++;
                streak++;
                showToast(`✓ Correct! +1 point`, false);
            } else {
                streak = 0;
                showToast(`✗ Try again! The sign is ${target.name}`, true);
            }
            document.getElementById('practiceScore').innerHTML = `
                <span>Score: ${score}</span>
                <span>Streak: ${streak} 🔥</span>
            `;
            currentIndex++;
        }, 3000);
    }, 5000);
}

// Initialize floating menu
function initFloatingMenu() {
    let fabOpen = false;
    fabMain.addEventListener('click', () => {
        fabOpen = !fabOpen;
        document.querySelector('.fab-options').classList.toggle('show', fabOpen);
    });
    
    document.querySelectorAll('.fab-option').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const action = btn.dataset.action;
            if (action === 'screenshot') takeScreenshot();
            if (action === 'share') shareProgress();
            if (action === 'export') exportData();
            if (action === 'feedback') showFeedbackForm();
        });
    });
}

function takeScreenshot() {
    if (video.videoWidth > 0) {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        canvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `signspeak-${Date.now()}.png`;
            a.click();
            showToast('Screenshot saved!', false);
        });
    }
}

function shareProgress() {
    const text = `I've learned ${totalGestures} signs with SignSpeak! My current streak is ${currentStreak} days! 🎉`;
    if (navigator.share) {
        navigator.share({ title: 'SignSpeak Progress', text: text });
    } else {
        navigator.clipboard.writeText(text);
        showToast('Progress copied to clipboard!', false);
    }
}

function exportData() {
    const data = {
        stats: { totalGestures, currentStreak, bestStreak },
        history: JSON.parse(localStorage.getItem('historyData') || '[]'),
        settings: { theme: document.documentElement.getAttribute('data-theme') }
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `signspeak-data-${Date.now()}.json`;
    a.click();
    showToast('Data exported successfully!', false);
}

function showFeedbackForm() {
    const feedback = prompt('Share your feedback to help us improve:');
    if (feedback) {
        showToast('Thank you for your feedback! 🙏', false);
        // Save feedback
        const feedbacks = JSON.parse(localStorage.getItem('feedbacks') || '[]');
        feedbacks.push({ text: feedback, date: new Date().toISOString() });
        localStorage.setItem('feedbacks', JSON.stringify(feedbacks));
    }
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
    openStatsBtn.addEventListener('click', () => statsModal.classList.add('active'));
    document.querySelectorAll('.modal-close, .modal-close-stats').forEach(btn => {
        btn.addEventListener('click', () => {
            settingsModal.classList.remove('active');
            statsModal.classList.remove('active');
        });
    });
    themeToggle.addEventListener('click', () => settingsModal.classList.add('active'));
    dayNightToggle.addEventListener('click', cycleTheme);
    voiceCmdBtn.addEventListener('click', toggleVoiceCommands);
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`${tabId}Tab`).classList.add('active');
        });
    });
    
    // Search and filter
    const librarySearch = document.getElementById('librarySearch');
    const categoryFilter = document.getElementById('categoryFilter');
    if (librarySearch) {
        librarySearch.addEventListener('input', (e) => {
            populateGesturesGrid(e.target.value, categoryFilter?.value || 'all');
        });
    }
    if (categoryFilter) {
        categoryFilter.addEventListener('change', (e) => {
            populateGesturesGrid(librarySearch?.value || '', e.target.value);
        });
    }
    
    thresholdSlider.addEventListener('input', (e) => {
        confidenceThreshold = e.target.value / 100;
        thresholdValue.textContent = `${e.target.value}%`;
    });
    
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            if (detectionModeSelect) detectionModeSelect.value = currentMode;
            showToast(`Switched to ${currentMode} mode`, false);
        });
    });
    
    detectionModeSelect?.addEventListener('change', (e) => {
        currentMode = e.target.value;
        modeBtns.forEach(btn => {
            if (btn.dataset.mode === currentMode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    });
    
    cameraSelect.addEventListener('change', async () => {
        if (isDetecting) await restartStream();
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space') {
            e.preventDefault();
            if (isDetecting) stopDetection();
            else startDetection();
        } else if (e.code === 'KeyS') {
            settingsModal.classList.add('active');
        } else if (e.code === 'KeyV') {
            toggleVoiceCommands();
        } else if (e.code === 'KeyD') {
            cycleTheme();
        } else if (e.code === 'KeyH') {
            showHelp();
        } else if (e.code === 'KeyT') {
            const nextTab = (Array.from(tabBtns).findIndex(b => b.classList.contains('active')) + 1) % tabBtns.length;
            tabBtns[nextTab].click();
        }
    });
    
    // Copy transcription
    document.getElementById('copyTranscriptionBtn')?.addEventListener('click', () => {
        const text = document.querySelector('.transcription-text')?.innerText;
        if (text && text !== 'Waiting for gestures...') {
            navigator.clipboard.writeText(text);
            showToast('Transcription copied!', false);
        }
    });
}

function showHelp() {
    const commands = [
        'Space - Start/Stop Detection',
        'T - Switch Tabs',
        'V - Voice Commands',
        'D - Change Theme',
        'S - Settings',
        'H - Show Help'
    ];
    showToast(commands.join(' | '), false);
}

function cycleTheme() {
    const themes = ['dark', 'light', 'neon', 'ocean'];
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = themes[(themes.indexOf(current) + 1) % themes.length];
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    showToast(`${next.charAt(0).toUpperCase() + next.slice(1)} mode activated`, false);
}

function loadThemePreference() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    if (themeModeSelect) themeModeSelect.value = savedTheme;
}

// Voice commands
function initVoiceCommands() {
    if ('webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const command = event.results[event.results.length - 1][0].transcript.toLowerCase();
            processVoiceCommand(command);
        };
        
        recognition.onerror = () => updateVoiceStatus(false);
        recognition.onend = () => updateVoiceStatus(false);
    }
}

function toggleVoiceCommands() {
    if (!recognition) {
        showToast('Voice commands not supported', true);
        return;
    }
    
    if (isListening) {
        recognition.stop();
        updateVoiceStatus(false);
    } else {
        recognition.start();
        updateVoiceStatus(true);
    }
}

function updateVoiceStatus(isActive) {
    isListening = isActive;
    const statusDiv = document.getElementById('voiceStatus');
    if (isActive) {
        statusDiv.innerHTML = '<i class="fas fa-microphone"></i> Listening...';
        voiceCmdBtn.classList.add('listening');
    } else {
        statusDiv.innerHTML = '<i class="fas fa-microphone-slash"></i> Click mic to start';
        voiceCmdBtn.classList.remove('listening');
    }
}

function processVoiceCommand(command) {
    showToast(`Command: "${command}"`, false);
    
    if (command.includes('start')) startDetection();
    else if (command.includes('stop')) stopDetection();
    else if (command.includes('clear')) clearHistory();
    else if (command.includes('screenshot')) takeScreenshot();
    else if (command.includes('share')) shareProgress();
    else if (command.includes('help')) showHelp();
    else if (command.includes('mode')) {
        const modes = ['standard', 'advanced', 'batch', 'precise'];
        const nextMode = modes[(modes.indexOf(currentMode) + 1) % modes.length];
        currentMode = nextMode;
        modeBtns.forEach(btn => {
            if (btn.dataset.mode === currentMode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
        showToast(`Switched to ${currentMode} mode`, false);
    }
}

// Start detection
async function startDetection() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined }
        });
        video.srcObject = stream;
        await video.play();
        
        updateConnectionStatus('connected', 'Active');
        isDetecting = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Detecting...';
        sessionStartTime = Date.now();
        
        startMockDetection();
    } catch (error) {
        showToast('Could not access camera', true);
    }
}

// Mock detection for demo
function startMockDetection() {
    if (frameInterval) clearInterval(frameInterval);
    frameInterval = setInterval(() => {
        if (!isDetecting) return;
        const randomGesture = gesturesDB[Math.floor(Math.random() * gesturesDB.length)];
        const confidence = 0.75 + Math.random() * 0.2;
        
        sentence.push(randomGesture.name.toLowerCase());
        if (sentence.length > 5) sentence.shift();
        
        updateUI({
            prediction: randomGesture.name.toLowerCase(),
            confidence: confidence,
            sentence: sentence,
            confidence_scores: Array(5).fill(0).map(() => Math.random())
        });
        
        totalGestures++;
        saveStats();
    }, 2000);
}

function updateUI(data) {
    const { prediction, confidence, sentence: newSentence, confidence_scores } = data;
    
    // Update transcription
    if (newSentence && newSentence.length > 0) {
        const transcriptionText = newSentence.map(g => {
            const gesture = gesturesDB.find(gd => gd.name.toLowerCase() === g);
            return gesture ? `${gesture.emoji} ${gesture.name}` : g;
        }).join(' → ');
        document.querySelector('.transcription-text').innerHTML = transcriptionText;
        
        if (voiceEnabled && newSentence[newSentence.length - 1] !== lastSpokenGesture) {
            speakText(`Detected ${newSentence[newSentence.length - 1]}`);
            lastSpokenGesture = newSentence[newSentence.length - 1];
            setTimeout(() => { lastSpokenGesture = ''; }, 2000);
        }
        
        addToHistory(newSentence[newSentence.length - 1]);
    }
    
    // Update confidence bars
    if (confidence_scores) {
        const container = document.getElementById('confidenceBars');
        const topGestures = gesturesDB.slice(0, 5);
        container.innerHTML = topGestures.map((gesture, idx) => {
            const score = confidence_scores[idx] || 0;
            return `
                <div class="confidence-item">
                    <span class="label">${gesture.emoji} ${gesture.name}</span>
                    <div class="bar-container">
                        <div class="bar" style="width: ${score * 100}%"></div>
                    </div>
                    <span class="value">${Math.round(score * 100)}%</span>
                </div>
            `;
        }).join('');
        
        // Update chart
        if (charts.confidence && charts.confidence.data) {
            charts.confidence.data.labels.push(new Date().toLocaleTimeString());
            charts.confidence.data.datasets[0].data.push(confidence);
            if (charts.confidence.data.labels.length > 20) {
                charts.confidence.data.labels.shift();
                charts.confidence.data.datasets[0].data.shift();
            }
            charts.confidence.update();
        }
    }
    
    const badge = document.getElementById('predictionBadge');
    if (badge && prediction) {
        const gesture = gesturesDB.find(g => g.name.toLowerCase() === prediction);
        badge.innerHTML = `
            <span class="prediction-label">${gesture?.emoji || '🤟'} ${gesture?.name || prediction}</span>
            <span class="prediction-confidence">${Math.round(confidence * 100)}%</span>
        `;
    }
    
    // Update FPS
    if (statFps) statFps.textContent = Math.floor(30 + Math.random() * 20);
}

function addToHistory(gesture) {
    const historyList = document.getElementById('historyList');
    const gestureData = gesturesDB.find(g => g.name.toLowerCase() === gesture);
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
        <span>${gestureData?.emoji || '🤟'} ${gestureData?.name || gesture}</span>
        <span>${new Date().toLocaleTimeString()}</span>
    `;
    historyList.insertBefore(historyItem, historyList.firstChild);
    while (historyList.children.length > 20) historyList.removeChild(historyList.lastChild);
    const placeholder = historyList.querySelector('.history-placeholder');
    if (placeholder) placeholder.remove();
}

function clearHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '<div class="history-placeholder">No predictions yet</div>';
    sentence = [];
    document.querySelector('.transcription-text').innerHTML = 'Waiting for gestures...';
    showToast('History cleared!', false);
}

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
    
    const badge = document.getElementById('predictionBadge');
    if (badge) badge.innerHTML = `<span class="prediction-label">Ready</span><span class="prediction-confidence">0%</span>`;
}

async function restartStream() {
    if (isDetecting) {
        stopDetection();
        await startDetection();
    }
}

function updateConnectionStatus(status, message) {
    if (connectionStatus) {
        connectionStatus.className = `connection-status ${status}`;
        connectionStatus.innerHTML = `<i class="fas fa-circle"></i> ${message}`;
    }
}

function speakText(text) {
    if (!voiceEnabled && voiceFeedbackToggle) voiceEnabled = voiceFeedbackToggle.checked;
    if (!voiceEnabled) return;
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 0.9;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);
    }
}

function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.style.borderLeftColor = isError ? '#ef4444' : '#10b981';
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}

function updateFPS() {
    let frameCount = 0;
    let lastTime = performance.now();
    function countFPS() {
        frameCount++;
        const now = performance.now();
        if (now - lastTime >= 1000) {
            if (statFps) statFps.textContent = frameCount;
            frameCount = 0;
            lastTime = now;
        }
        requestAnimationFrame(countFPS);
    }
    countFPS();
}

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

document.addEventListener('mousemove', (e) => {
    const glow = document.querySelector('.cursor-glow');
    if (glow) glow.style.transform = `translate(${e.clientX - 200}px, ${e.clientY - 200}px)`;
});

if (voiceFeedbackToggle) {
    voiceFeedbackToggle.addEventListener('change', (e) => {
        voiceEnabled = e.target.checked;
    });
}

init();
