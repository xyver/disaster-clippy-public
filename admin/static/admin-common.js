/**
 * Admin Common JS - Shared functionality for all admin pages
 */

// ============================================================================
// Mode Toggle - Local/Global test mode switching
// ============================================================================

// Server's default mode from VECTOR_DB_MODE env var (fetched on init)
let serverDefaultMode = 'local';

// Get current effective mode (toggle override or server default)
function getTestMode() {
    const toggleOverride = localStorage.getItem('adminModeOverride');
    // If toggle has been used, return that; otherwise return server default
    if (toggleOverride !== null) {
        return toggleOverride;
    }
    return serverDefaultMode;
}

// Check if in global test mode
function isGlobalTestMode() {
    return getTestMode() === 'global';
}

// Update UI based on current mode
function updateModeUI() {
    const isGlobal = isGlobalTestMode();
    const toggle = document.getElementById('modeToggle');
    const label = document.getElementById('modeLabel');
    const checkbox = document.getElementById('modeCheckbox');
    const badge = document.getElementById('modeBadge');

    if (!toggle || !label || !checkbox) return;

    checkbox.checked = isGlobal;
    toggle.classList.toggle('global', isGlobal);
    label.textContent = isGlobal ? 'Global' : 'Local';

    // Show TEST badge only when toggle is overriding the server default
    const toggleOverride = localStorage.getItem('adminModeOverride');
    if (badge) {
        const isOverriding = toggleOverride !== null && toggleOverride !== serverDefaultMode;
        badge.style.display = isOverriding ? 'inline' : 'none';
    }

    // Update CSS to match current mode
    updateAdminCSS(isGlobal);

    // Update global-only features visibility
    updateGlobalFeatures();
}

// Dynamically switch between local-admin.css and global-admin.css
function updateAdminCSS(isGlobal) {
    // Find the admin CSS link element
    const cssLinks = document.querySelectorAll('link[rel="stylesheet"]');
    for (const link of cssLinks) {
        const href = link.getAttribute('href');
        if (href && href.includes('-admin.css')) {
            const newCss = isGlobal ? 'global-admin.css' : 'local-admin.css';
            const newHref = href.replace(/(?:local|global)-admin\.css/, newCss);
            if (link.getAttribute('href') !== newHref) {
                link.setAttribute('href', newHref);
                console.log('Switched admin CSS to:', newCss);
            }
            break;
        }
    }
}

// Show/hide global-only features based on mode
function updateGlobalFeatures() {
    const isGlobal = isGlobalTestMode();

    // Pinecone page is global-only for write operations
    const pineconeLink = document.querySelector('a[href="/useradmin/pinecone"]');
    if (pineconeLink) {
        pineconeLink.style.opacity = isGlobal ? '1' : '0.5';
        pineconeLink.title = isGlobal ? 'Pinecone (full access)' : 'Pinecone (read-only in local mode)';
    }

    // Dispatch event for page-specific handlers
    window.dispatchEvent(new CustomEvent('modeUIUpdated', { detail: { isGlobal } }));
}

// Toggle between local and global mode (temporary override for testing)
function toggleMode() {
    const checkbox = document.getElementById('modeCheckbox');
    const newMode = checkbox.checked ? 'global' : 'local';

    // Store as override
    localStorage.setItem('adminModeOverride', newMode);
    updateModeUI();

    // Dispatch event for other components to react
    window.dispatchEvent(new CustomEvent('testModeChanged', { detail: { mode: newMode, isGlobal: checkbox.checked } }));
    console.log('Test mode override set to:', newMode, '(server default:', serverDefaultMode, ')');
}

// Clear the toggle override and use server default
function resetToServerDefault() {
    localStorage.removeItem('adminModeOverride');
    updateModeUI();
    console.log('Reset to server default mode:', serverDefaultMode);
}

// Fetch and apply server's default mode from VECTOR_DB_MODE
async function initServerMode() {
    try {
        const resp = await fetch('/useradmin/api/admin-mode');
        const data = await resp.json();
        serverDefaultMode = data.mode || 'local';
        console.log('Server VECTOR_DB_MODE:', serverDefaultMode);
        updateModeUI();
    } catch (e) {
        console.error('Could not fetch admin mode, defaulting to local:', e);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

// Format file size for display
function formatFileSize(bytes) {
    if (!bytes) return '-';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

// Format duration in seconds to human readable
function formatDuration(seconds) {
    if (!seconds) return '';
    if (seconds < 60) return Math.round(seconds) + 's';
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return mins + 'm ' + secs + 's';
}

// Format date for display
function formatDate(dateString) {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString();
}

// Show a toast notification
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        container.style.cssText = 'position: fixed; bottom: 20px; right: 20px; z-index: 9999;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    const colors = {
        'info': '#0f3460',
        'success': '#1a5c3a',
        'warning': '#5c4a1a',
        'error': '#5c1a1a'
    };
    toast.style.cssText = 'background: ' + (colors[type] || colors.info) +
        '; color: #eee; padding: 0.75rem 1rem; border-radius: 4px; margin-top: 0.5rem; ' +
        'box-shadow: 0 2px 8px rgba(0,0,0,0.3); animation: fadeIn 0.3s;';
    toast.textContent = message;
    container.appendChild(toast);

    // Remove after 3 seconds
    setTimeout(function() {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(function() { toast.remove(); }, 300);
    }, 3000);
}

// Confirm dangerous action with double confirmation
async function confirmDangerousAction(message, confirmText) {
    if (!confirm(message)) return false;
    const input = prompt('Type "' + confirmText + '" to confirm:');
    return input === confirmText;
}

// ============================================================================
// API Helper Functions
// ============================================================================

// Generic fetch wrapper with error handling
async function apiFetch(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.detail || 'Request failed');
        }

        return data;
    } catch (e) {
        console.error('API Error:', e);
        throw e;
    }
}

// ============================================================================
// Initialization
// ============================================================================

// Initialize: fetch server mode then update UI
async function initAdmin() {
    await initServerMode();
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initAdmin();
});

// Also initialize immediately if DOM is already loaded
if (document.readyState !== 'loading') {
    initAdmin();
}
