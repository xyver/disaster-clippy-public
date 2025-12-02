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
        const isZimUrl = url.startsWith('zim://');

        // For ZIM URLs, show title without link; for real URLs, make it clickable
        const titleHtml = isZimUrl
            ? `<span class="zim-title">${escapeHtml(article.title)}</span><span class="zim-badge">offline</span>`
            : `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(article.title)}</a>`;

        return `
            <div class="article-card">
                <h3>
                    <span>#${idx + 1}</span>
                    ${titleHtml}
                    <span class="score-badge">${(article.score * 100).toFixed(0)}% match</span>
                </h3>
                <div class="article-meta">
                    Source: ${escapeHtml(article.source)}
                </div>
                <div class="article-snippet">${escapeHtml(article.snippet)}</div>
            </div>
        `;
    }).join('');
}

// Send chat message
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
        if (selectedSources !== null && selectedSources.length > 0) {
            requestBody.sources = selectedSources;
        }

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        // Store session ID for conversation continuity
        sessionId = data.session_id;

        // Add assistant response
        addMessage(data.response);

        // Update articles panel
        renderArticles(data.articles);

    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again.');
    } finally {
        loading.classList.remove('active');
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Parse markdown to HTML (links, bold, italic, headers)
function parseMarkdown(text) {
    if (!text) return '';

    // First escape HTML to prevent XSS
    let html = escapeHtml(text);

    // Convert headers (### Header -> <h4>, ## Header -> <h3>, # Header -> <h2>)
    html = html.replace(/^### (.+)$/gm, '<h4 class="md-header">$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3 class="md-header">$1</h3>');
    html = html.replace(/^# (.+)$/gm, '<h2 class="md-header">$1</h2>');

    // Convert markdown links [text](url) to clickable links
    // Only allow http, https URLs for safety
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // Convert bold **text** to <strong>
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Convert italic *text* to <em> (but not inside URLs or already processed)
    html = html.replace(/(?<![*])\*([^*]+)\*(?![*])/g, '<em>$1</em>');

    // Convert numbered lists (lines starting with number.)
    html = html.replace(/^(\d+)\.\s+/gm, '<span class="list-number">$1.</span> ');

    return html;
}

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
    if (selectedSources === null || selectedSources.length === Object.keys(availableSources).length) {
        toggleSourcesBtn.textContent = 'Select Sources (All)';
    } else if (selectedSources.length === 0) {
        toggleSourcesBtn.textContent = 'Select Sources (None)';
    } else {
        toggleSourcesBtn.textContent = `Select Sources (${selectedSources.length})`;
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

// Initialize
loadWelcome();
loadSources();
userInput.focus();
