/**
 * Shared visualization JavaScript for Knowledge Map
 * Used by both public (/visualise) and admin (/useradmin/visualise) pages
 *
 * Requires: IS_ADMIN and API_PREFIX to be set before this script loads
 */

// State
let visData = null;
let edgesBySource = {};   // Per-source edges: {source_id: {edges: [...]}}
let urlsBySource = {};    // Per-source URLs: {source_id: {urls: {...}}}
let loadingEdges = {};    // Track in-flight edge requests per source
let loadingUrls = {};     // Track in-flight URL requests per source
let selectedSources = new Set();
let selectedLinkSources = new Set();
let sourceColorMap = {};
let pollingInterval = null;
let panelCollapsed = false;
let showEdges = false;  // Default OFF - edges are large and slow to render

// Color palette for sources
const colorPalette = [
    '#4a9eff', '#4ecca3', '#ff6b6b', '#ffd93d', '#6bcb77',
    '#9b59b6', '#e17055', '#00cec9', '#fd79a8', '#a29bfe',
    '#fab1a0', '#74b9ff', '#55a3ff', '#ff7675', '#fdcb6e'
];

// Panel toggle
function togglePanel() {
    panelCollapsed = !panelCollapsed;
    document.getElementById('panelContent').classList.toggle('collapsed', panelCollapsed);
    document.getElementById('panelToggle').textContent = panelCollapsed ? '+' : '-';
}

// Edge toggle - just toggles visibility, edges load per-source via toggleLink
function toggleEdges() {
    showEdges = document.getElementById('showEdgesToggle').checked;
    renderPlot();
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStatus();
    initDraggablePanel();
});

// Make panel draggable
function initDraggablePanel() {
    const panel = document.getElementById('controlPanel');
    const header = panel.querySelector('.panel-header');

    let isDragging = false;
    let offsetX = 0;
    let offsetY = 0;

    header.addEventListener('mousedown', (e) => {
        // Don't drag if clicking the toggle button
        if (e.target.classList.contains('panel-toggle')) return;

        isDragging = true;
        offsetX = e.clientX - panel.offsetLeft;
        offsetY = e.clientY - panel.offsetTop;
        header.style.cursor = 'grabbing';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        let newX = e.clientX - offsetX;
        let newY = e.clientY - offsetY;

        // Keep panel within viewport bounds
        const maxX = window.innerWidth - panel.offsetWidth - 10;
        const maxY = window.innerHeight - panel.offsetHeight - 10;
        newX = Math.max(10, Math.min(newX, maxX));
        newY = Math.max(10, Math.min(newY, maxY));

        panel.style.left = newX + 'px';
        panel.style.top = newY + 'px';
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        header.style.cursor = 'grab';
    });

    // Set initial cursor
    header.style.cursor = 'grab';
}

async function loadStatus() {
    try {
        const resp = await fetch(API_PREFIX + '/api/visualise/status');
        const status = await resp.json();

        updateStatusDisplay(status);

        if (status.job_active) {
            startPolling();
        } else if (status.has_data) {
            loadVisualisationData();
        }
    } catch (e) {
        console.error('Error loading status:', e);
        document.getElementById('statusValue').textContent = 'Error';
    }
}

function updateStatusDisplay(status) {
    const generateBtn = document.getElementById('generateBtn');
    const progressContainer = document.getElementById('progressContainer');

    if (status.job_active) {
        document.getElementById('statusValue').textContent = 'Generating...';
        document.getElementById('statusValue').style.color = '#f9a825';
        if (IS_ADMIN && generateBtn) {
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            if (progressContainer) {
                progressContainer.style.display = 'block';
                document.getElementById('progressFill').style.width = status.job_progress + '%';
                document.getElementById('progressText').textContent = status.job_message || 'Processing...';
            }
        }
    } else if (status.has_data) {
        document.getElementById('statusValue').textContent = 'Ready';
        document.getElementById('statusValue').style.color = '#4ecca3';
        if (IS_ADMIN && generateBtn) {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Regenerate';
            if (progressContainer) progressContainer.style.display = 'none';
        }
    } else {
        document.getElementById('statusValue').textContent = 'No Data';
        document.getElementById('statusValue').style.color = '#888';
        if (IS_ADMIN && generateBtn) {
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Visualization';
            if (progressContainer) progressContainer.style.display = 'none';
        }
    }

    document.getElementById('pointsValue').textContent = status.point_count ? status.point_count.toLocaleString() : '-';
    document.getElementById('sourcesValue').textContent = status.sources ? status.sources.length : '-';
    document.getElementById('varianceValue').textContent = status.variance_explained ?
        (status.variance_explained * 100).toFixed(1) + '%' : '-';
    document.getElementById('edgesValue').textContent = status.edge_count ? status.edge_count.toLocaleString() : '-';

    if (status.generated_at) {
        const date = new Date(status.generated_at);
        document.getElementById('generatedValue').textContent = formatRelativeTime(date);
    } else {
        document.getElementById('generatedValue').textContent = '-';
    }
}

function formatRelativeTime(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return diffMins + 'm ago';
    if (diffHours < 24) return diffHours + 'h ago';
    return diffDays + 'd ago';
}

async function generateVisualisation() {
    if (!IS_ADMIN) return;

    const generateBtn = document.getElementById('generateBtn');
    try {
        generateBtn.disabled = true;
        generateBtn.textContent = 'Starting...';

        const resp = await fetch(API_PREFIX + '/api/visualise/generate', { method: 'POST' });
        const result = await resp.json();

        if (result.status === 'started' || result.status === 'already_running') {
            startPolling();
        } else {
            alert('Error: ' + (result.message || 'Unknown error'));
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Visualization';
        }
    } catch (e) {
        console.error('Error starting generation:', e);
        alert('Error starting generation: ' + e.message);
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Visualization';
    }
}

function startPolling() {
    if (pollingInterval) return;

    const progressContainer = document.getElementById('progressContainer');
    if (IS_ADMIN && progressContainer) {
        progressContainer.style.display = 'block';
    }

    pollingInterval = setInterval(async () => {
        try {
            const resp = await fetch(API_PREFIX + '/api/visualise/status');
            const status = await resp.json();

            updateStatusDisplay(status);

            if (!status.job_active) {
                stopPolling();
                if (status.has_data) {
                    loadVisualisationData();
                }
            }
        } catch (e) {
            console.error('Polling error:', e);
        }
    }, 1000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function loadVisualisationData() {
    try {
        document.getElementById('loadingOverlay').classList.remove('hidden');
        document.getElementById('loadingText').textContent = 'Loading visualization data...';
        document.getElementById('emptyState').style.display = 'none';

        const resp = await fetch(API_PREFIX + '/api/visualise/data');
        if (!resp.ok) {
            throw new Error('Failed to load data');
        }

        visData = await resp.json();

        // Reset per-source lazy-loaded data
        edgesBySource = {};
        urlsBySource = {};
        loadingEdges = {};
        loadingUrls = {};

        // Build source-to-color mapping FIRST (consistent across all UI)
        sourceColorMap = {};
        visData.sources.forEach((source, idx) => {
            sourceColorMap[source] = colorPalette[idx % colorPalette.length];
        });

        // Initialize selected sources (all selected)
        selectedSources = new Set(visData.sources);
        selectedLinkSources = new Set();  // Start with no links visible

        // Ensure edge toggle is unchecked by default
        const edgeToggle = document.getElementById('showEdgesToggle');
        if (edgeToggle) {
            edgeToggle.checked = false;
            showEdges = false;
        }

        // Build source filter UI
        buildSourceFilters();
        buildLinkFilters();

        // Render plot
        renderPlot();

        document.getElementById('loadingOverlay').classList.add('hidden');
    } catch (e) {
        console.error('Error loading visualization data:', e);
        document.getElementById('loadingOverlay').classList.add('hidden');
        document.getElementById('emptyState').style.display = 'flex';
    }
}

function buildSourceFilters() {
    const sourceList = document.getElementById('sourceList');
    const counts = visData.source_counts || {};

    let html = '';
    visData.sources.forEach((source, idx) => {
        const count = counts[source] || 0;
        const checked = selectedSources.has(source) ? 'checked' : '';
        const color = sourceColorMap[source];

        html += `
            <div class="source-item">
                <input type="checkbox" id="src_${idx}" value="${source}" ${checked}
                       onchange="toggleSource('${source}')" style="accent-color: ${color}">
                <label for="src_${idx}" style="color: ${color}">${source}</label>
                <span class="source-count">(${count.toLocaleString()})</span>
            </div>
        `;
    });

    sourceList.innerHTML = html;
}

function toggleSource(source) {
    if (selectedSources.has(source)) {
        selectedSources.delete(source);
    } else {
        selectedSources.add(source);
    }
    renderPlot();
}

function selectAllSources() {
    sourceColorMap = {};
    visData.sources.forEach((source, idx) => {
        sourceColorMap[source] = colorPalette[idx % colorPalette.length];
    });

    selectedSources = new Set(visData.sources);
    selectedLinkSources = new Set();

    buildSourceFilters();
    buildLinkFilters();
    renderPlot();
}

function selectNoneSources() {
    selectedSources = new Set();
    buildSourceFilters();
    buildLinkFilters();
    renderPlot();
}

function buildLinkFilters() {
    if (!visData || !visData.sources) return;

    const linkList = document.getElementById('linkList');
    const counts = visData.source_counts || {};

    let html = '';
    visData.sources.forEach((source, idx) => {
        const count = counts[source] || 0;
        const checked = selectedLinkSources.has(source) ? 'checked' : '';
        const color = sourceColorMap[source];

        html += `
            <div class="source-item">
                <input type="checkbox" id="link_${idx}" value="${source}" ${checked}
                       onchange="toggleLink('${source}')" style="accent-color: ${color}">
                <label for="link_${idx}" style="color: ${color}">${source}</label>
                <span class="source-count">(links)</span>
            </div>
        `;
    });

    linkList.innerHTML = html;
}

async function toggleLink(source) {
    if (selectedLinkSources.has(source)) {
        selectedLinkSources.delete(source);
        renderPlot();
    } else {
        selectedLinkSources.add(source);

        // Lazy load edges for this source if not yet loaded
        if (!edgesBySource[source] && !loadingEdges[source]) {
            loadingEdges[source] = true;
            try {
                const resp = await fetch(API_PREFIX + '/api/visualise/edges/' + encodeURIComponent(source));
                if (resp.ok) {
                    edgesBySource[source] = await resp.json();
                    console.log('Loaded edges for', source, ':', edgesBySource[source].edge_count || 0);
                } else {
                    edgesBySource[source] = { edges: [] };
                }
            } catch (e) {
                console.warn('Could not load edges for', source, ':', e);
                edgesBySource[source] = { edges: [] };
            }
            loadingEdges[source] = false;
        }

        renderPlot();
    }
}

function selectAllLinks() {
    selectedLinkSources = new Set(visData.sources);
    buildLinkFilters();
    renderPlot();
}

function selectNoneLinks() {
    selectedLinkSources = new Set();
    buildLinkFilters();
    renderPlot();
}

function renderPlot() {
    if (!visData || !visData.points) return;

    // Filter points by selected sources
    const filteredPoints = visData.points.filter(p => selectedSources.has(p.source));

    // Update point count overlay
    document.getElementById('pointCountOverlay').style.display = filteredPoints.length > 0 ? 'block' : 'none';
    document.getElementById('displayedPoints').textContent = filteredPoints.length.toLocaleString();

    if (filteredPoints.length === 0) {
        document.getElementById('emptyState').style.display = 'flex';
        document.getElementById('emptyState').innerHTML = `
            <h2>No Points Selected</h2>
            <p>Select at least one source from the control panel.</p>
        `;
        Plotly.purge('plotContainer');
        return;
    }

    document.getElementById('emptyState').style.display = 'none';

    // Build index mapping for filtered points
    const pointIndexMap = new Map();
    filteredPoints.forEach((p, newIdx) => {
        const origIdx = visData.points.indexOf(p);
        pointIndexMap.set(origIdx, newIdx);
    });

    // Group points by source for coloring
    const traces = [];
    visData.sources.forEach((source, idx) => {
        const sourcePoints = filteredPoints.filter(p => p.source === source);
        if (sourcePoints.length === 0) return;

        const color = sourceColorMap[source];

        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: source,
            x: sourcePoints.map(p => p.x),
            y: sourcePoints.map(p => p.y),
            z: sourcePoints.map(p => p.z),
            text: sourcePoints.map(p => p.title),
            customdata: sourcePoints.map(p => ({
                id: p.id,
                url: p.url,
                local_url: p.local_url,
                doc_type: p.doc_type
            })),
            hovertemplate: '<b>%{text}</b><br>Source: ' + source + '<extra></extra>',
            marker: {
                size: 3,
                color: color,
                opacity: 0.8
            }
        });
    });

    // Add edges (connection lines) if enabled - gather from per-source data
    if (showEdges && selectedLinkSources.size > 0) {
        const edgeX = [];
        const edgeY = [];
        const edgeZ = [];

        // Gather edges from all selected sources that have been loaded
        selectedLinkSources.forEach(source => {
            const sourceEdges = edgesBySource[source]?.edges || [];
            sourceEdges.forEach(edge => {
                if (pointIndexMap.has(edge.from) && pointIndexMap.has(edge.to)) {
                    const fromPoint = visData.points[edge.from];
                    const toPoint = visData.points[edge.to];
                    edgeX.push(fromPoint.x, toPoint.x, null);
                    edgeY.push(fromPoint.y, toPoint.y, null);
                    edgeZ.push(fromPoint.z, toPoint.z, null);
                }
            });
        });

        if (edgeX.length > 0) {
            traces.unshift({
                type: 'scatter3d',
                mode: 'lines',
                name: 'Links',
                x: edgeX,
                y: edgeY,
                z: edgeZ,
                line: {
                    color: 'rgba(100, 100, 100, 0.3)',
                    width: 1
                },
                hoverinfo: 'none',
                showlegend: false
            });
        }
    }

    const layout = {
        paper_bgcolor: '#1a1a2e',
        plot_bgcolor: '#1a1a2e',
        font: { color: '#ccc' },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
            xaxis: {
                title: '',
                showticklabels: false,
                gridcolor: '#2a2a4a',
                zerolinecolor: '#3a3a5a'
            },
            yaxis: {
                title: '',
                showticklabels: false,
                gridcolor: '#2a2a4a',
                zerolinecolor: '#3a3a5a'
            },
            zaxis: {
                title: '',
                showticklabels: false,
                gridcolor: '#2a2a4a',
                zerolinecolor: '#3a3a5a'
            },
            bgcolor: '#1a1a2e'
        },
        legend: {
            x: 1,
            y: 0.5,
            xanchor: 'right',
            bgcolor: 'rgba(22, 33, 62, 0.9)',
            bordercolor: '#0f3460',
            borderwidth: 1
        },
        showlegend: true
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
        displaylogo: false
    };

    Plotly.newPlot('plotContainer', traces, layout, config);

    // Click handler to show popup
    document.getElementById('plotContainer').on('plotly_click', function(data) {
        const point = data.points[0];
        if (point) {
            showDocPopup(point, data.event);
        }
    });
}

async function showDocPopup(point, event) {
    const popup = document.getElementById('docPopup');
    const title = point.text || 'Untitled';
    const source = point.data.name || 'unknown';

    document.getElementById('popupTitle').textContent = title;
    document.getElementById('popupMeta').textContent = 'Source: ' + source;

    const linkEl = document.getElementById('popupLink');

    // Get URL - check customdata first, then lazy-load from per-source URLs endpoint
    let url = point.customdata?.url || point.customdata?.local_url || '';

    // If no URL in customdata, try to load from per-source URLs file
    if (!url) {
        // Find original point index to look up URL
        const origIdx = visData.points.findIndex(p =>
            p.x === point.x && p.y === point.y && p.z === point.z
        );

        if (origIdx >= 0) {
            // Lazy load URLs for this source if not yet loaded
            if (!urlsBySource[source] && !loadingUrls[source]) {
                loadingUrls[source] = true;
                try {
                    linkEl.textContent = 'Loading...';
                    linkEl.style.display = 'inline-block';
                    linkEl.removeAttribute('href');

                    const resp = await fetch(API_PREFIX + '/api/visualise/urls/' + encodeURIComponent(source));
                    if (resp.ok) {
                        urlsBySource[source] = await resp.json();
                    } else {
                        urlsBySource[source] = { urls: {} };
                    }
                } catch (e) {
                    console.warn('Could not load URLs for', source, ':', e);
                    urlsBySource[source] = { urls: {} };
                }
                loadingUrls[source] = false;
            }

            // Look up URL by point index
            if (urlsBySource[source]?.urls) {
                url = urlsBySource[source].urls[origIdx] || '';
            }
        }
    }

    if (url) {
        linkEl.href = url;
        linkEl.textContent = 'Open Article';
        linkEl.style.display = 'inline-block';
    } else {
        linkEl.style.display = 'none';
    }

    // Position popup near click, but keep on screen
    let x, y;
    if (event && event.clientX !== undefined) {
        x = event.clientX + 10;
        y = event.clientY + 10;
    } else {
        x = window.innerWidth / 2 - 175;
        y = window.innerHeight / 2 - 60;
    }

    const popupWidth = 350;
    const popupHeight = 120;
    if (event && event.clientX !== undefined) {
        if (x + popupWidth > window.innerWidth) {
            x = event.clientX - popupWidth - 10;
        }
        if (y + popupHeight > window.innerHeight) {
            y = event.clientY - popupHeight - 10;
        }
    } else {
        if (x < 0) x = 10;
        if (y < 0) y = 10;
        if (x + popupWidth > window.innerWidth) x = window.innerWidth - popupWidth - 10;
        if (y + popupHeight > window.innerHeight) y = window.innerHeight - popupHeight - 10;
    }

    popup.style.left = x + 'px';
    popup.style.top = y + 'px';
    popup.classList.remove('hidden');
}

function closePopup() {
    document.getElementById('docPopup').classList.add('hidden');
}

// Close popup when clicking elsewhere
document.addEventListener('click', function(e) {
    const popup = document.getElementById('docPopup');
    if (!popup.contains(e.target) && !e.target.closest('.js-plotly-plot')) {
        closePopup();
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    if (visData && visData.points) {
        Plotly.Plots.resize('plotContainer');
    }
});