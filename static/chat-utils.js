/**
 * Shared chat utility functions for Disaster Clippy
 * Used by both main chat (chat.js) and embed widget (embed-widget.html)
 */

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
    // Allow http, https URLs (external - opens in new tab)
    // Regex handles URLs with parentheses like Wikipedia: Residency_(medicine)
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^()\s]*(?:\([^()]*\)[^()\s]*)*)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // Also allow local /zim/ URLs (opens in new tab to preserve chat history)
    html = html.replace(/\[([^\]]+)\]\((\/zim\/[^()\s]*(?:\([^()]*\)[^()\s]*)*)\)/g,
        '<a href="$2" target="_blank" class="zim-link">$1</a>');

    // Also allow local /backup/ URLs (HTML backup server)
    html = html.replace(/\[([^\]]+)\]\((\/backup\/[^()\s]*(?:\([^()]*\)[^()\s]*)*)\)/g,
        '<a href="$2" target="_blank" class="backup-link">$1</a>');

    // Convert bold **text** to <strong>
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Convert italic *text* to <em> (but not inside URLs or already processed)
    html = html.replace(/(?<![*])\*([^*]+)\*(?![*])/g, '<em>$1</em>');

    // Convert numbered lists (lines starting with number.)
    html = html.replace(/^(\d+)\.\s+/gm, '<span class="list-number">$1.</span> ');

    // Convert line breaks to <br> for proper display
    html = html.replace(/\n/g, '<br>');

    return html;
}
