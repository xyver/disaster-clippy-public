/**
 * Source Card - Unified rendering for source cards across admin pages
 *
 * Usage:
 *   const html = SourceCard.render(source, options);
 *
 * Options:
 *   variant: 'sources' | 'cloud' | 'packs' - determines which fields to show
 *   showStatusBoxes: true | false - show 6-item validation status boxes
 *   showChecklist: true | false - show 4-item checklist (cloud style)
 *   onInstall: function(sourceId, includeBackup) - install button handler
 *   onEdit: function(sourceId) - edit button handler
 *   onUpload: function(sourceId) - upload button handler
 */

const SourceCard = (function() {
    'use strict';

    // =============================================================================
    // UTILITY FUNCTIONS
    // =============================================================================

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatNumber(num) {
        if (!num) return '0';
        return num.toLocaleString();
    }

    // =============================================================================
    // STATUS DETERMINATION
    // =============================================================================

    /**
     * Determine the card's overall status class
     */
    function getCardStatusClass(source, options) {
        if (options.variant === 'cloud') {
            return source.is_complete ? 'source-card--complete' : 'source-card--incomplete';
        }

        // Sources page logic
        if (source.status === 'installed') {
            return 'source-card--complete';
        } else if (source.status === 'available') {
            return 'source-card--available';
        } else if (source.status === 'local_only') {
            // Check if source is complete enough
            const hasEssentials = source.has_config && source.has_metadata && source.has_backup;
            return hasEssentials ? '' : 'source-card--incomplete';
        }
        return '';
    }

    /**
     * Get badge text and class based on source status
     */
    function getBadgeInfo(source, options) {
        if (options.variant === 'cloud') {
            // Cloud page: use can_publish for global admin, can_submit for local admin
            if (source.can_publish) {
                return { text: 'Ready to Publish', class: 'source-card__badge--ready' };
            } else if (source.can_submit) {
                return { text: 'Ready to Submit', class: 'source-card__badge--available' };
            } else if (source.is_complete) {
                // Basic files present, but not validated yet
                return { text: 'Needs Validation', class: 'source-card__badge--local' };
            } else {
                return { text: 'Incomplete', class: 'source-card__badge--incomplete' };
            }
        }

        // Sources page logic - use new gates when available
        if (source.can_publish) {
            return { text: 'Production Ready', class: 'source-card__badge--installed' };
        } else if (source.can_submit) {
            return { text: 'Ready to Submit', class: 'source-card__badge--available' };
        }

        // Fallback to status-based badges
        switch (source.status) {
            case 'installed':
                return { text: 'Installed', class: 'source-card__badge--installed' };
            case 'available':
                return { text: 'Available', class: 'source-card__badge--available' };
            case 'local_only':
            default:
                // Check if basic files are present
                if (source.is_complete) {
                    return { text: 'Ready', class: 'source-card__badge--available' };
                }
                return { text: 'Local Only', class: 'source-card__badge--local' };
        }
    }

    // =============================================================================
    // RENDER COMPONENTS
    // =============================================================================

    /**
     * Render the card header with name and badge
     */
    function renderHeader(source, options) {
        const badge = getBadgeInfo(source, options);
        const name = source.name || source.source_id || 'Unknown';

        return `
            <div class="source-card__header">
                <h3 class="source-card__name">${escapeHtml(name)}</h3>
                <span class="source-card__badge ${badge.class}">${badge.text}</span>
            </div>
        `;
    }

    /**
     * Render metadata row (document count, backup type, license, etc.)
     */
    function renderMeta(source, options) {
        const parts = [];

        // Document count
        const docCount = source.document_count || source.total_docs || 0;
        parts.push(`<span>${formatNumber(docCount)} docs</span>`);

        // Backup info
        if (source.backup_type) {
            parts.push(`<span>${source.backup_type} backup</span>`);
        } else if (source.backup_size_mb) {
            parts.push(`<span>${source.backup_size_mb} MB</span>`);
        }

        // License
        if (options.variant !== 'cloud') {
            const license = source.license && source.license !== 'Unknown'
                ? source.license
                : 'Unknown license';
            parts.push(`<span>${license}</span>`);
        }

        // Last published (cloud page)
        if (options.variant === 'cloud' && source.last_published) {
            parts.push(`<span class="source-card__meta--highlight">Published: ${formatPublishDate(source.last_published)}</span>`);
        }

        return `<div class="source-card__meta">${parts.join('')}</div>`;
    }

    /**
     * Format a publish date to relative time
     */
    function formatPublishDate(isoString) {
        if (!isoString) return '';
        try {
            const date = new Date(isoString);
            const now = new Date();
            const diffMs = now - date;
            const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

            if (diffDays === 0) return 'today';
            if (diffDays === 1) return 'yesterday';
            if (diffDays < 7) return `${diffDays} days ago`;
            return date.toLocaleDateString();
        } catch (e) {
            return isoString.split('T')[0];
        }
    }

    /**
     * Render status boxes (sources page style)
     * Shows validation status for all required checks
     */
    function renderStatusBoxes(source) {
        const hasConfig = source.has_config !== false;
        const hasBackup = source.has_backup !== false;
        const hasMetadata = source.has_metadata === true;
        const has1536 = source.has_vectors_1536 === true;
        const has768 = source.has_vectors_768 === true;
        const licenseOk = source.license_in_allowlist === true;
        const licenseVerified = source.license_verified === true;
        const linksOffline = source.links_verified_offline === true;
        const linksOnline = source.links_verified_online === true;
        const langEnglish = source.language_is_english === true;

        // Core requirements (must pass for can_submit)
        const boxes = [
            { label: 'Config', status: hasConfig ? 'ok' : 'missing' },
            { label: 'Backup', status: hasBackup ? 'ok' : 'missing' },
            { label: 'Meta', status: hasMetadata ? 'ok' : 'missing' },
            { label: '1536', status: has1536 ? 'ok' : 'warning' },
            { label: '768', status: has768 ? 'ok' : 'warning' },
            { label: 'License', status: (licenseOk && licenseVerified) ? 'ok' : (licenseOk ? 'warning' : 'missing') },
            { label: 'Links', status: (linksOffline && linksOnline) ? 'ok' : 'warning' },
            { label: 'Lang', status: langEnglish ? 'ok' : 'warning' }
        ];

        const boxHtml = boxes.map(box =>
            `<span class="source-card__status-box source-card__status-box--${box.status}">${box.label}</span>`
        ).join('');

        return `<div class="source-card__status-boxes">${boxHtml}</div>`;
    }

    /**
     * Render checklist (cloud page style)
     * Shows all validation requirements for publishing
     */
    function renderChecklist(source) {
        const items = [
            { label: 'Config', has: source.has_config },
            { label: 'Metadata', has: source.has_metadata },
            { label: 'Backup', has: source.has_backup },
            { label: '1536 Vectors', has: source.has_vectors_1536 },
            { label: '768 Vectors', has: source.has_vectors_768 },
            { label: 'License OK', has: source.license_in_allowlist },
            { label: 'License Verified', has: source.license_verified },
            { label: 'Links Verified', has: source.links_verified_offline && source.links_verified_online },
            { label: 'English', has: source.language_is_english },
            { label: 'Lang Verified', has: source.language_verified }
        ];

        const itemHtml = items.map(item => {
            const statusClass = item.has ? 'source-card__check-item--has' : 'source-card__check-item--missing';
            const icon = item.has ? '[x]' : '[ ]';
            return `<span class="source-card__check-item ${statusClass}">${icon} ${item.label}</span>`;
        }).join('');

        return `<div class="source-card__checklist">${itemHtml}</div>`;
    }

    /**
     * Render missing items warning
     */
    function renderMissing(source) {
        if (!source.missing || source.missing.length === 0) return '';
        return `<div class="source-card__missing">Missing: ${source.missing.join(', ')}</div>`;
    }

    /**
     * Render action buttons
     */
    function renderActions(source, options) {
        const buttons = [];

        if (options.variant === 'cloud') {
            // Cloud upload page buttons
            if (source.is_complete) {
                const isUploaded = options.uploadedSources && options.uploadedSources.has(source.source_id);
                const isGlobal = options.isGlobalAdmin;
                const disabled = !options.cloudConnected || isUploaded;

                let btnText = isUploaded
                    ? (isGlobal ? 'Published' : 'Submitted')
                    : (isGlobal ? 'Publish to Production' : 'Submit for Review');

                let btnClass = isUploaded ? 'source-card__btn--uploaded' : 'source-card__btn--primary';

                buttons.push(`
                    <button class="source-card__btn ${btnClass}"
                            data-action="upload"
                            data-source-id="${source.source_id}"
                            ${disabled ? 'disabled' : ''}>
                        ${btnText}
                    </button>
                `);
            } else {
                buttons.push(`
                    <button class="source-card__btn source-card__btn--primary" disabled>
                        Complete requirements first
                    </button>
                `);
            }
        } else {
            // Sources page buttons
            if (source.is_local) {
                buttons.push(`
                    <button class="source-card__btn source-card__btn--edit"
                            data-action="edit"
                            data-source-id="${source.source_id}">
                        Edit
                    </button>
                `);
                if (source.is_cloud) {
                    buttons.push(`<span class="source-card__helper-text source-card__helper-text--success">(from cloud)</span>`);
                }
            } else if (source.is_cloud) {
                buttons.push(`
                    <button class="source-card__btn source-card__btn--install"
                            data-action="install"
                            data-source-id="${source.source_id}"
                            data-include-backup="false"
                            title="Download vectors only for search">
                        Index Only
                    </button>
                `);
                buttons.push(`
                    <button class="source-card__btn source-card__btn--install-full"
                            data-action="install"
                            data-source-id="${source.source_id}"
                            data-include-backup="true"
                            title="Download vectors + backup files for offline viewing">
                        Full Install
                    </button>
                `);
            }
        }

        return `<div class="source-card__actions">${buttons.join('')}</div>`;
    }

    /**
     * Render progress bar (hidden by default)
     */
    function renderProgress(source) {
        return `
            <div class="source-card__progress" id="progress-${source.source_id}">
                <div class="source-card__progress-fill"></div>
            </div>
        `;
    }

    // =============================================================================
    // MAIN RENDER FUNCTION
    // =============================================================================

    /**
     * Render a complete source card
     *
     * @param {Object} source - Source data object
     * @param {Object} options - Rendering options
     * @returns {string} HTML string for the card
     */
    function render(source, options = {}) {
        // Default options
        options = {
            variant: 'sources',  // 'sources', 'cloud', 'packs'
            showStatusBoxes: true,
            showChecklist: false,
            cloudConnected: true,
            isGlobalAdmin: false,
            uploadedSources: null,
            ...options
        };

        // Determine what to show based on variant
        if (options.variant === 'cloud') {
            options.showStatusBoxes = false;
            options.showChecklist = true;
        } else if (options.variant === 'sources') {
            // Only show status boxes for local sources
            options.showStatusBoxes = source.status !== 'available';
            options.showChecklist = false;
        }

        const cardClass = getCardStatusClass(source, options);

        let html = `<div class="source-card ${cardClass}" data-source-id="${source.source_id}">`;

        // Header (name + badge)
        html += renderHeader(source, options);

        // Metadata row
        html += renderMeta(source, options);

        // Status indicators
        if (options.showStatusBoxes) {
            html += renderStatusBoxes(source);
        }
        if (options.showChecklist) {
            html += renderChecklist(source);
        }

        // Missing items warning
        if (options.variant === 'cloud') {
            html += renderMissing(source);
        }

        // Action buttons
        html += renderActions(source, options);

        // Progress bar
        html += renderProgress(source);

        html += '</div>';

        return html;
    }

    /**
     * Render multiple cards into a grid
     */
    function renderGrid(sources, options = {}) {
        if (!sources || sources.length === 0) {
            const message = options.emptyMessage || 'No sources found.';
            return `<div class="source-card-placeholder">${message}</div>`;
        }

        const cards = sources.map(source => render(source, options)).join('');
        const gridClass = options.singleColumn ? 'source-card-grid source-card-grid--single' : 'source-card-grid';
        return `<div class="${gridClass}">${cards}</div>`;
    }

    /**
     * Attach event handlers to cards in a container
     */
    function attachHandlers(container, handlers = {}) {
        if (!container) return;

        container.addEventListener('click', function(e) {
            const btn = e.target.closest('[data-action]');
            if (!btn) return;

            const action = btn.dataset.action;
            const sourceId = btn.dataset.sourceId;

            if (action === 'edit' && handlers.onEdit) {
                handlers.onEdit(sourceId);
            } else if (action === 'install' && handlers.onInstall) {
                const includeBackup = btn.dataset.includeBackup === 'true';
                handlers.onInstall(sourceId, includeBackup, btn);
            } else if (action === 'upload' && handlers.onUpload) {
                handlers.onUpload(sourceId, btn);
            }
        });
    }

    /**
     * Update a card's progress bar
     */
    function updateProgress(sourceId, percent) {
        const progress = document.getElementById(`progress-${sourceId}`);
        if (progress) {
            progress.classList.add('active');
            const fill = progress.querySelector('.source-card__progress-fill');
            if (fill) {
                fill.style.width = percent + '%';
            }
        }
    }

    /**
     * Hide a card's progress bar
     */
    function hideProgress(sourceId) {
        const progress = document.getElementById(`progress-${sourceId}`);
        if (progress) {
            progress.classList.remove('active');
        }
    }

    // =============================================================================
    // PUBLIC API
    // =============================================================================

    return {
        render: render,
        renderGrid: renderGrid,
        attachHandlers: attachHandlers,
        updateProgress: updateProgress,
        hideProgress: hideProgress,
        escapeHtml: escapeHtml
    };

})();

// Export for module systems if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SourceCard;
}
