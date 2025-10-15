// Global state
let currentSessionId = null;
let messageCount = 0;
let isProcessing = false;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadStatus = document.getElementById('uploadStatus');
const fileInfo = document.getElementById('fileInfo');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const sessionBadge = document.getElementById('sessionBadge');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatInputContainer = document.getElementById('chatInputContainer');
const loadingOverlay = document.getElementById('loadingOverlay');
const messageCountEl = document.getElementById('messageCount');
const docCountEl = document.getElementById('docCount');

// ==================== FILE UPLOAD HANDLERS ====================
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files).slice(0, 2); // Limit to 2 files
    if (files.length > 0) uploadFiles(files);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).slice(0, 2); // Limit to 2 files
    if (files.length > 0) uploadFiles(files);
});

// Upload multiple files function
async function uploadFiles(files) {
    const formData = new FormData();
    files.forEach((file, idx) => {
        formData.append('files', file);
    });
    showLoading(true);
    showUploadStatus('Uploading and indexing documents...', 'info');
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok && data.indexed) {
            currentSessionId = data.session_id;
            updateSessionBadge(currentSessionId);
            showUploadStatus(data.message, 'success');
            displayFileInfo(files.map(f => f.name).join(', '));
            enableChat();
            updateStats(files.length, 0);
        } else {
            showUploadStatus('Error uploading file: ' + (data.detail || 'Unknown error'), 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus('Error uploading file: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

// Show upload status
function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            uploadStatus.style.display = 'none';
        }, 5000);
    }
}

// Display file info
function displayFileInfo(fileName) {
    fileInfo.innerHTML = `
        <h4><i class="fas fa-file-alt"></i> Uploaded Document</h4>
        <p>${fileName}</p>
    `;
    fileInfo.classList.add('visible');
}

// Update session badge
function updateSessionBadge(sessionId) {
    sessionBadge.textContent = `Session: ${sessionId}`;
    sessionBadge.classList.add('active');
}

// Enable chat
function enableChat() {
    welcomeScreen.style.display = 'none';
    chatMessages.style.display = 'block';
    chatInputContainer.style.display = 'block';
}

// ==================== CHAT HANDLERS ====================
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener('input', () => {
    // Auto-resize textarea
    chatInput.style.height = 'auto';
    chatInput.style.height = chatInput.scrollHeight + 'px';
});

async function sendMessage() {
    const message = chatInput.value.trim();
    
    if (!message || !currentSessionId || isProcessing) return;
    
    isProcessing = true;
    sendBtn.disabled = true;
    
    // Add user message to chat
    addMessage('user', message);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Show typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            addMessage('assistant', data.answer, data.source_documents);
            updateStats(null, messageCount + 2); // +2 for user + assistant
        } else {
            addMessage('assistant', 'Error: ' + (data.detail || 'Failed to get response'));
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator(typingId);
        addMessage('assistant', 'Error: Failed to send message. Please try again.');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

// Add message to chat
function addMessage(role, text, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const icon = role === 'user' ? 'fa-user' : 'fa-robot';
    const label = role === 'user' ? 'You' : 'AI Assistant';
    
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        // Only show top 2 sources
        const topSources = sources.slice(0, 2);
        sourcesHtml = `
            <div class="sources">
                <div class="sources-header">
                    <i class="fas fa-book"></i> Sources
                </div>
                ${topSources.map((source, idx) => `
                    <div class="source-item">
                        <strong>Source ${idx + 1}:</strong> ${source.metadata?.filename || 'Document'}
                        ${source.content ? `<div class="source-content">${source.content}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-header">
                <i class="fas ${icon}"></i>
                <span>${label}</span>
            </div>
            <div class="message-text">${formatMessage(text)}</div>
            ${sourcesHtml}
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Format message text (handle line breaks, etc.)
function formatMessage(text) {
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

// Add typing indicator
function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'message assistant';
    
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="message-header">
                <i class="fas fa-robot"></i>
                <span>AI Assistant</span>
            </div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingId;
}

// Remove typing indicator
function removeTypingIndicator(typingId) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
}

// ==================== UTILITY FUNCTIONS ====================
function showLoading(show) {
    loadingOverlay.classList.toggle('active', show);
}

function updateStats(docs, messages) {
    if (docs !== null) docCountEl.textContent = docs;
    if (messages !== null) {
        messageCount = messages;
        messageCountEl.textContent = messages;
    }
}

function resetSession() {
    if (confirm('Start a new session? This will clear the current chat.')) {
        currentSessionId = null;
        messageCount = 0;
        chatMessages.innerHTML = '';
        chatMessages.style.display = 'none';
        chatInputContainer.style.display = 'none';
        welcomeScreen.style.display = 'flex';
        fileInfo.classList.remove('visible');
        sessionBadge.textContent = 'No Session';
        sessionBadge.classList.remove('active');
        fileInput.value = '';
        updateStats(0, 0);
    }
}

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    console.log('Multi-Doc RAG Chat initialized');
    chatInput.focus();
});
