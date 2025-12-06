// Disaster Clippy - Chat Interface

let sessionId = null;
let availableSources = {};  // {source_id: {name, count}}
let selectedSources = null; // null = all sources, array = specific sources

const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const articlesList = document.getElementById('articlesList');
const loading = document.getElementById('loading');
const indexStats = document.getElementById('indexStats');
const sourcesPanel = document.getElementById('sourcesPanel');
const sourcesGrid = document.getElementById('sourcesGrid');
const toggleSourcesBtn = document.getElementById('toggleSources');
const selectAllBtn = document.getElementById('selectAll');
const selectNoneBtn = document.getElementById('selectNone');

// Load welcome message and stats on page load
async function loadWelcome() {
    try {
        const response = await fetch('/welcome');
        const data = await response.json();

        // Update stats bar
        const stats = data.stats;
        if (stats.total_documents > 0) {
            const topicsStr = stats.topics.length > 0 ? ` | Topics: ${stats.topics.join(', ')}` : '';
            indexStats.textContent = `${stats.total_documents} articles indexed${topicsStr}`;
        } else {
            indexStats.textContent = 'No articles indexed yet';
        }

        // Update welcome message in chat
        const welcomeDiv = chatMessages.querySelector('.message.assistant .message-content');
        if (welcomeDiv && data.message) {
            welcomeDiv.textContent = data.message;
        }

    } catch (e) {
        console.error('Failed to load welcome:', e);
        indexStats.textContent = 'Unable to load stats';
    }
}

// Add message to chat
function addMessage(content, isUser = false) {
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user' : 'assistant'}`;
    // Use parseMarkdown for assistant messages (may contain links), escapeHtml for user
    const formattedContent = isUser ? escapeHtml(content) : parseMarkdown(content);
    div.innerHTML = `<div class="message-content">${formattedContent}</div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Render articles panel
function renderArticles(articles) {
    if (!articles || articles.length === 0) {
        articlesList.innerHTML = '<div class="empty-state">No matching articles found</div>';
        return;
    }

    articlesList.innerHTML = articles.map((article, idx) => {
        const url = article.url || '';
        const isLocalZim = url.startsWith('/zim/');
        const isExternalUrl = url.startsWith('http://') || url.startsWith('https://');

        // Build the title HTML with appropriate link
        let titleHtml;
        if (isLocalZim) {
            // Local ZIM URL - opens in new tab to preserve chat history
            titleHtml = `<a href="${escapeHtml(url)}" target="_blank" class="zim-link">${escapeHtml(article.title)}</a><span class="zim-badge">local</span>`;
        } else if (isExternalUrl) {
            // External URL - opens in new tab
            titleHtml = `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(article.title)}</a>`;
        } else if (url.startsWith('zim://')) {
            // Fallback for unconverted zim:// URLs (should not happen)
            titleHtml = `<span class="zim-title">${escapeHtml(article.title)}</span><span class="zim-badge">offline</span>`;
        } else {
            // No URL or unknown format
            titleHtml = `<span>${escapeHtml(article.title)}</span>`;
        }

        return `
            <div class="article-card">
                <h3>
                    <span>#${idx + 1}</span>
                    ${titleHtml}
                    <span class="score-badge">${(article.score * 100).toFixed(0)}% match</span>
                </h3>
                <div class="article-meta">
                    Source: ${escapeHtml(article.source)}${isLocalZim ? ' (offline)' : ''}
                </div>
                <div class="article-snippet">${escapeHtml(article.snippet)}</div>
            </div>
        `;
    }).join('');
}

// Send chat message with streaming
async function sendMessage(message) {
    if (!message.trim()) return;

    // Add user message to chat
    addMessage(message, true);

    // Clear input
    userInput.value = '';

    // Show loading
    loading.classList.add('active');
    sendBtn.disabled = true;

    try {
        const requestBody = {
            message: message,
            session_id: sessionId
        };

        // Add source filter if not all sources selected
        if (selectedSources !== null) {
            requestBody.sources = selectedSources;
        }

        // Use streaming endpoint
        const response = await fetch('/api/v1/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // Create a placeholder for the streaming response
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        messageDiv.innerHTML = '<div class="message-content"></div>';
        chatMessages.appendChild(messageDiv);
        const contentDiv = messageDiv.querySelector('.message-content');

        // Hide loading once we start receiving
        loading.classList.remove('active');

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let buffer = '';  // Buffer to handle partial SSE messages

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Append new data to buffer
            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages (separated by \n\n)
            const messages = buffer.split('\n\n');

            // Keep the last part in buffer (may be incomplete)
            buffer = messages.pop() || '';

            for (const message of messages) {
                const lines = message.split('\n');
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);

                        if (data === '[DONE]') {
                            // Stream complete - parse markdown
                            contentDiv.innerHTML = parseMarkdown(fullResponse);
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (data.startsWith('[ARTICLES]')) {
                            // Parse articles JSON
                            try {
                                const articlesJson = data.substring(10);
                                const articles = JSON.parse(articlesJson);
                                renderArticles(articles);
                            } catch (e) {
                                console.error('Failed to parse articles:', e);
                            }
                        } else if (data.startsWith('[ERROR]')) {
                            fullResponse += 'Error: ' + data.substring(7);
                        } else {
                            // Regular text chunk - unescape newlines
                            const text = data.replace(/\\n/g, '\n');
                            fullResponse += text;
                            // Update display with escaped HTML (will parse markdown at end)
                            contentDiv.textContent = fullResponse;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        }
                    }
                }
            }
        }

    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again.');
        loading.classList.remove('active');
    } finally {
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// escapeHtml() and parseMarkdown() are loaded from chat-utils.js

// Event listeners
chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage(userInput.value);
});

// Allow Enter to submit (Shift+Enter for newline)
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(userInput.value);
    }
});

// Load sources and render checkboxes
async function loadSources() {
    try {
        const response = await fetch('/sources');
        const data = await response.json();

        availableSources = data.sources || {};

        // Load saved preferences from localStorage
        const saved = localStorage.getItem('clippy_selected_sources');
        if (saved) {
            try {
                const savedSources = JSON.parse(saved);
                // Only use saved sources that still exist
                selectedSources = savedSources.filter(s => s in availableSources);
                if (selectedSources.length === 0) {
                    selectedSources = null; // All sources if none selected
                }
            } catch (e) {
                selectedSources = null;
            }
        }

        renderSourcesGrid();
        updateToggleButton();

    } catch (e) {
        console.error('Failed to load sources:', e);
        sourcesGrid.innerHTML = '<span style="color: #888;">Unable to load sources</span>';
    }
}

// Render the sources checkboxes
function renderSourcesGrid() {
    const sourceIds = Object.keys(availableSources).sort();

    if (sourceIds.length === 0) {
        sourcesGrid.innerHTML = '<span style="color: #888;">No sources indexed yet</span>';
        return;
    }

    sourcesGrid.innerHTML = sourceIds.map(sourceId => {
        const source = availableSources[sourceId];
        const isChecked = selectedSources === null || selectedSources.includes(sourceId);
        const displayName = source.name || sourceId;

        return `
            <div class="source-item">
                <input type="checkbox" id="source-${sourceId}" value="${sourceId}"
                       ${isChecked ? 'checked' : ''} onchange="onSourceChange()">
                <label for="source-${sourceId}">
                    ${escapeHtml(displayName)}
                    <span class="source-count">(${source.count})</span>
                </label>
            </div>
        `;
    }).join('');
}

// Handle source checkbox changes
function onSourceChange() {
    const checkboxes = sourcesGrid.querySelectorAll('input[type="checkbox"]');
    const checked = [];
    let allChecked = true;

    checkboxes.forEach(cb => {
        if (cb.checked) {
            checked.push(cb.value);
        } else {
            allChecked = false;
        }
    });

    // If all are checked, set to null (query all)
    selectedSources = allChecked ? null : checked;

    // Save to localStorage
    if (selectedSources === null) {
        localStorage.removeItem('clippy_selected_sources');
    } else {
        localStorage.setItem('clippy_selected_sources', JSON.stringify(selectedSources));
    }

    updateToggleButton();
}

// Update the toggle button text
function updateToggleButton() {
    const totalSources = Object.keys(availableSources).length;
    if (selectedSources === null || selectedSources.length === totalSources) {
        toggleSourcesBtn.textContent = `Select Sources (${totalSources}/${totalSources})`;
    } else if (selectedSources.length === 0) {
        toggleSourcesBtn.textContent = `Select Sources (0/${totalSources})`;
    } else {
        toggleSourcesBtn.textContent = `Select Sources (${selectedSources.length}/${totalSources})`;
    }
}

// Toggle sources panel
function toggleSourcesPanel() {
    sourcesPanel.classList.toggle('open');
}

// Select all sources
function selectAllSources() {
    const checkboxes = sourcesGrid.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = true);
    selectedSources = null;
    localStorage.removeItem('clippy_selected_sources');
    updateToggleButton();
}

// Select no sources
function selectNoSources() {
    const checkboxes = sourcesGrid.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = false);
    selectedSources = [];
    localStorage.setItem('clippy_selected_sources', JSON.stringify([]));
    updateToggleButton();
}

// Event listeners for source controls
toggleSourcesBtn.addEventListener('click', toggleSourcesPanel);
selectAllBtn.addEventListener('click', selectAllSources);
selectNoneBtn.addEventListener('click', selectNoSources);

// Connection status - uses unified endpoint
async function loadConnectionStatus() {
    try {
        const response = await fetch('/api/v1/connection-status');
        const data = await response.json();

        const dot = document.getElementById('connectionDot');
        const label = document.getElementById('connectionLabel');
        const container = document.getElementById('connectionStatus');

        if (dot && label) {
            // Map state to CSS class
            const stateClass = data.state || 'online';
            dot.className = 'connection-dot ' + stateClass;
            label.textContent = data.state_label || 'Unknown';

            // Update tooltip with full message
            if (container && data.message) {
                container.title = data.message;
            }
        }
    } catch (e) {
        console.error('Failed to load connection status:', e);
        const label = document.getElementById('connectionLabel');
        if (label) {
            label.textContent = 'Error';
        }
    }
}

// Refresh connection status periodically (every 30 seconds)
setInterval(loadConnectionStatus, 30000);

// Initialize
loadWelcome();
loadSources();
loadConnectionStatus();
userInput.focus();
