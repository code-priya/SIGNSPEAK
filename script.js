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
        canvas.toBl
