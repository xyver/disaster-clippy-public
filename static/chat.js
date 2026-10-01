// Disaster Clippy - Chat Interface

function createSessionId() {
    return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

let sessionId = createSessionId();
let availableSources = {};  // {source_id: {name, count}}
let selectedSources = null; // Saved selection used by chat; null = all sources
let draftSources = null;    // Checkbox selection awaiting Save
let welcomeData = null;
let activeChatController = null;
let chatGeneration = 0;

function readableList(items) {
    if (items.length < 2) return items[0] || '';
    if (items.length === 2) return `${items[0]} and ${items[1]}`;
    return `${items.slice(0, -1).join(', ')}, and ${items.at(-1)}`;
}

function selectedCollectionNames(ids) {
    const names = ids.map(id => availableSources[id].short_name || availableSources[id].name || id);
    if (names.length > 4) return `${names.slice(0, 3).join(', ')}, and ${names.length - 3} more`;
    return readableList(names);
}

const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const loading = document.getElementById('loading');
const indexStats = document.getElementById('indexStats');
const sourcesPanel = document.getElementById('sourcesPanel');
const sourcesGrid = document.getElementById('sourcesGrid');
const toggleSourcesBtn = document.getElementById('toggleSources');
const selectAllBtn = document.getElementById('selectAll');
const selectNoneBtn = document.getElementById('selectNone');
const saveSourcesBtn = document.getElementById('saveSources');
const sourcesSaveStatus = document.getElementById('sourcesSaveStatus');

function selectedSourceIds() {
    const ids = Object.keys(availableSources);
    return selectedSources === null ? ids : selectedSources.filter(id => id in availableSources);
}

function renderCollectionContext() {
    const allIds = Object.keys(availableSources);
    const chosen = selectedSourceIds();
    const total = chosen.reduce((sum, id) => sum + (Number(availableSources[id].count) || 0), 0);
    let opening;

    if (allIds.length === 0) {
        const stats = welcomeData?.stats || {};
        indexStats.textContent = stats.total_documents > 0
            ? `${stats.total_documents} articles indexed`
            : 'Loading collection...';
        opening = welcomeData?.message || 'Ask what you need. I will search the collection and show the sources.';
    } else if (chosen.length === 0) {
        indexStats.textContent = 'No collections selected';
        opening = 'Choose at least one collection, then select Save collection to start a new chat.';
    } else if (selectedSources === null) {
        const topics = welcomeData?.stats?.topics || [];
        indexStats.textContent = `${total} passages in collection${topics.length ? ` | Topics: ${topics.join(', ')}` : ''}`;
        opening = `You are searching all ${allIds.length} collections (${total.toLocaleString('en-US')} passages)${topics.length ? `, covering ${readableList(topics)}` : ''}. Ask what you need and check the references in each answer.`;
    } else if (chosen.length === 1) {
        const name = availableSources[chosen[0]].name || chosen[0];
        indexStats.textContent = `${total} passages selected`;
        opening = `You are searching ${name} (${total} passages). Ask a question about this collection; I will cite the sources I use.`;
    } else {
        indexStats.textContent = `${total} passages across ${chosen.length} selected collections`;
        opening = `You are searching ${chosen.length} collections: ${selectedCollectionNames(chosen)} (${total.toLocaleString('en-US')} passages). Ask what you need and check the references in each answer.`;
    }

    if (chatMessages.children.length === 1) {
        const openingDiv = chatMessages.querySelector('.message.assistant .message-content');
        if (openingDiv) openingDiv.textContent = opening;
    }
    sendBtn.disabled = activeChatController !== null || (chosen.length === 0 && allIds.length > 0);
}

// Load welcome message and stats on page load
async function loadWelcome() {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    try {
        const response = await fetch('/welcome', { signal: controller.signal });
        if (!response.ok) throw new Error(`Welcome request failed: ${response.status}`);
        welcomeData = await response.json();
        renderCollectionContext();
    } catch (e) {
        console.error('Failed to load welcome:', e);
        renderCollectionContext();
    } finally {
        clearTimeout(timeoutId);
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

// Attach reference cards to a specific message div
function renderArticles(articles, messageDiv) {
    if (!articles || articles.length === 0) return;

    const refsSection = document.createElement('div');
    refsSection.className = 'message-references';

    const label = document.createElement('p');
    label.className = 'references-label';
    label.textContent = 'References';
    refsSection.appendChild(label);

    const cardsDiv = document.createElement('div');
    cardsDiv.className = 'reference-cards';

    cardsDiv.innerHTML = articles.map((article, idx) => {
        const url = article.url || '';
        const isLocalZim = url.startsWith('/zim/');
        const isLocalBackup = url.startsWith('/backup/');
        const isExternalUrl = url.startsWith('http://') || url.startsWith('https://');

        let titleHtml;
        if (isLocalZim) {
            titleHtml = `<a href="${escapeHtml(url)}" target="_blank" class="zim-link">${escapeHtml(article.title)}</a><span class="zim-badge">local</span>`;
        } else if (isLocalBackup) {
            titleHtml = `<a href="${escapeHtml(url)}" target="_blank" class="backup-link">${escapeHtml(article.title)}</a><span class="zim-badge">local</span>`;
        } else if (isExternalUrl) {
            titleHtml = `<a href="${escapeHtml(url)}" target="_blank">${escapeHtml(article.title)}</a>`;
        } else if (url.startsWith('zim://')) {
            titleHtml = `<span class="zim-title">${escapeHtml(article.title)}</span><span class="zim-badge">offline</span>`;
        } else {
            titleHtml = `<span>${escapeHtml(article.title)}</span>`;
        }

        return `
            <div class="article-card">
                <div class="article-card-header">
                    <span class="cite-num">${idx + 1}</span>
                    <h3>${titleHtml}</h3>
                </div>
                <div class="article-meta">${escapeHtml(article.source)}${(isLocalZim || isLocalBackup) ? ' \u00b7 offline' : ''}</div>
                <div class="article-snippet">${escapeHtml(article.snippet)}</div>
            </div>
        `;
    }).join('');

    refsSection.appendChild(cardsDiv);
    messageDiv.appendChild(refsSection);
}

// Send chat message with streaming
async function sendMessage(message) {
    if (!message.trim() || activeChatController || (selectedSources !== null && selectedSources.length === 0)) return;
    const generation = chatGeneration;
    const controller = new AbortController();
    activeChatController = controller;

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

        // Add search language for localized sources (Phase 4)
        const searchLanguage = localStorage.getItem('searchLanguage') || 'en';
        if (searchLanguage && searchLanguage !== 'en') {
            requestBody.search_language = searchLanguage;
        }

        // Use streaming endpoint
        const response = await fetch('/api/v1/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal
        });

        if (generation !== chatGeneration) return;

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
        let pendingArticles = null;  // Hold articles until text is done streaming

        while (true) {
            const { done, value } = await reader.read();
            if (generation !== chatGeneration) {
                await reader.cancel();
                return;
            }
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
                            // Stream complete - parse markdown, then attach references
                            contentDiv.innerHTML = parseMarkdown(fullResponse);
                            if (pendingArticles) {
                                renderArticles(pendingArticles, messageDiv);
                            }
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (data.startsWith('[ARTICLES]')) {
                            // Store articles — render after text finishes so they appear below
                            try {
                                const articlesJson = data.substring(10);
                                pendingArticles = JSON.parse(articlesJson);
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
        if (error.name === 'AbortError' || generation !== chatGeneration) return;
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request. Please try again.');
        loading.classList.remove('active');
    } finally {
        if (activeChatController === controller) activeChatController = null;
        if (generation === chatGeneration) {
            renderCollectionContext();
            userInput.focus();
        }
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
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    try {
        const response = await fetch('/sources', { signal: controller.signal });
        if (!response.ok) {
            throw new Error(`Sources request failed: ${response.status}`);
        }
        const data = await response.json();

        availableSources = data.sources || {};

        // Restore only the last saved selection. Checkbox edits are drafts.
        const saved = localStorage.getItem('clippy_selected_sources');
        if (saved !== null) {
            try {
                const savedSources = JSON.parse(saved);
                if (!Array.isArray(savedSources)) throw new Error('Invalid saved collection');
                const validSources = savedSources.filter(id => id in availableSources);
                selectedSources = validSources.length === Object.keys(availableSources).length
                    ? null : validSources;
            } catch (e) {
                selectedSources = null;
            }
        }
        draftSources = selectedSources === null ? null : [...selectedSources];

        renderSourcesGrid();
        updateToggleButton();
        updateSaveButton();
        renderCollectionContext();

    } catch (e) {
        console.error('Failed to load sources:', e);
        if (e.name === 'AbortError') {
            sourcesGrid.innerHTML = '<span style="color: #f0ad4e;">Sources are taking longer than expected. Retrying...</span>';
            // Retry after 30 seconds
            setTimeout(loadSources, 30000);
        } else {
            sourcesGrid.innerHTML = '<span style="color: #888;">Unable to load sources</span>';
        }
    } finally {
        clearTimeout(timeoutId);
    }
}

// Render the sources checkboxes with offline/online indicators
function renderSourcesGrid() {
    const sourceIds = Object.keys(availableSources).sort();

    if (sourceIds.length === 0) {
        sourcesGrid.innerHTML = '<span style="color: #888;">No sources indexed yet</span>';
        return;
    }

    sourcesGrid.innerHTML = sourceIds.map(sourceId => {
        const source = availableSources[sourceId];
        const isChecked = draftSources === null || draftSources.includes(sourceId);
        const displayName = source.name || sourceId;

        // Check availability - 768-dim means offline ready
        const has768 = source.has_768 === true;
        const has1536 = source.has_1536 === true;

        // Build availability indicator
        let badge = '';
        let badgeTitle = '';

        if (has768 && has1536) {
            badge = '<span class="source-badge both" title="Available online and offline">both</span>';
            badgeTitle = 'Available online and offline';
        } else if (has768) {
            badge = '<span class="source-badge offline" title="Offline only (768-dim)">offline</span>';
            badgeTitle = 'Offline only';
        } else if (has1536) {
            badge = '<span class="source-badge online" title="Online only (needs 768-dim for offline)">online</span>';
            badgeTitle = 'Online only - needs 768-dim for offline';
        }

        return `
            <div class="source-item" title="${badgeTitle}">
                <input type="checkbox" id="source-${sourceId}" value="${sourceId}"
                       ${isChecked ? 'checked' : ''} onchange="onSourceChange()">
                <label for="source-${sourceId}">
                    ${escapeHtml(displayName)}
                    <span class="source-count">(${source.count})</span>
                    ${badge}
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

    draftSources = allChecked ? null : checked;
    updateSaveButton();
}

function updateSaveButton() {
    const saved = selectedSources === null ? null : [...selectedSources].sort();
    const draft = draftSources === null ? null : [...draftSources].sort();
    const changed = JSON.stringify(saved) !== JSON.stringify(draft);
    saveSourcesBtn.disabled = !changed;
    sourcesSaveStatus.textContent = changed ? 'Unsaved collection changes' : '';
}

function saveCollection() {
    if (saveSourcesBtn.disabled) return;
    selectedSources = draftSources === null ? null : [...draftSources];
    if (selectedSources === null) {
        localStorage.removeItem('clippy_selected_sources');
    } else {
        localStorage.setItem('clippy_selected_sources', JSON.stringify(selectedSources));
    }

    // Start a new conversation for the newly saved source set.
    chatGeneration += 1;
    activeChatController?.abort();
    activeChatController = null;
    sessionId = createSessionId();
    chatMessages.innerHTML = '<div class="message assistant"><div class="message-content"></div></div>';
    loading.classList.remove('active');
    userInput.value = '';
    renderCollectionContext();
    updateToggleButton();
    updateSaveButton();
    sourcesPanel.classList.remove('open');
    toggleSourcesBtn.setAttribute('aria-expanded', 'false');
    userInput.focus();
}

// Update the toggle button text
function updateToggleButton() {
    const totalSources = Object.keys(availableSources).length;
    if (selectedSources === null || selectedSources.length === totalSources) {
        toggleSourcesBtn.textContent = `Collection (${totalSources})`;
    } else if (selectedSources.length === 0) {
        toggleSourcesBtn.textContent = `Collection (0/${totalSources})`;
    } else {
        toggleSourcesBtn.textContent = `Collection (${selectedSources.length}/${totalSources})`;
    }
}

// Toggle sources panel
function toggleSourcesPanel() {
    sourcesPanel.classList.toggle('open');
    toggleSourcesBtn.setAttribute('aria-expanded', sourcesPanel.classList.contains('open'));
}

// Select all sources
function selectAllSources() {
    const checkboxes = sourcesGrid.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = true);
    draftSources = null;
    updateSaveButton();
}

// Select no sources
function selectNoSources() {
    const checkboxes = sourcesGrid.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = false);
    draftSources = [];
    updateSaveButton();
}

// Event listeners for source controls
toggleSourcesBtn.addEventListener('click', toggleSourcesPanel);
selectAllBtn.addEventListener('click', selectAllSources);
selectNoneBtn.addEventListener('click', selectNoSources);
saveSourcesBtn.addEventListener('click', saveCollection);

// Connection status - uses unified endpoint
async function loadConnectionStatus() {
    try {
        // Use AbortController for timeout - database may be busy during indexing
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch('/api/v1/connection-status', { signal: controller.signal });
        clearTimeout(timeoutId);

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
        const dot = document.getElementById('connectionDot');
        const label = document.getElementById('connectionLabel');
        if (dot) dot.className = 'connection-dot offline';
        if (label) {
            label.textContent = e.name === 'AbortError' ? 'Busy' : 'Error';
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
