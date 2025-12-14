/**
 * Admin Common JS - Shared functionality for all admin pages
 */

// ============================================================================
// Admin Mode - Based on VECTOR_DB_MODE env var (no toggle override)
// ============================================================================

// Server mode from VECTOR_DB_MODE env var (fetched on init)
let serverMode = 'local';

// Check if in global admin mode (based on VECTOR_DB_MODE env)
function isGlobalTestMode() {
    return serverMode === 'global';
}

// Alias for clarity
function isGlobalAdmin() {
    return serverMode === 'global';
}

// Get current mode
function getAdminMode() {
    return serverMode;
}

// Update UI based on current mode
function updateModeUI() {
    const isGlobal = isGlobalAdmin();

    // Update mode indicator in nav
    const indicator = document.getElementById('adminModeIndicator');
    if (indicator) {
        indicator.textContent = isGlobal ? 'Global' : 'Local';
        indicator.classList.toggle('global', isGlobal);
    }

    // Update global-only features visibility
    updateGlobalFeatures();

    // Dispatch event for page-specific handlers
    window.dispatchEvent(new CustomEvent('adminModeLoaded', { detail: { mode: serverMode, isGlobal } }));
}

// Show/hide global-only features based on mode
function updateGlobalFeatures() {
    const isGlobal = isGlobalAdmin();

    // Pinecone page is global-only for write operations
    const pineconeLink = document.querySelector('a[href="/useradmin/pinecone"]');
    if (pineconeLink) {
        pineconeLink.style.opacity = isGlobal ? '1' : '0.5';
        pineconeLink.title = isGlobal ? 'Pinecone (full access)' : 'Pinecone (read-only in local mode)';
    }
}

// Fetch server mode from VECTOR_DB_MODE
async function initServerMode() {
    try {
        const resp = await fetch('/useradmin/api/admin-mode');
        const data = await resp.json();
        // Handle both response formats (app.py returns mode, source_tools returns is_global_admin)
        serverMode = data.mode || (data.is_global_admin ? 'global' : 'local');
        console.log('Admin mode (VECTOR_DB_MODE):', serverMode);
        updateModeUI();
    } catch (e) {
        console.error('Could not fetch admin mode, defaulting to local:', e);
        serverMode = 'local';
        updateModeUI();
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
