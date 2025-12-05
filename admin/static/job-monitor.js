/**
 * Job Monitor - Shared component for tracking background jobs
 * Include this script in any admin page to show active job status
 */

class JobMonitor {
    constructor() {
        this.pollInterval = 2000; // Poll every 2 seconds when jobs active
        this.idleInterval = 10000; // Poll every 10 seconds when idle
        this.activeJobs = [];
        this.callbacks = {};
        this.pollingTimer = null;
        this.container = null;

        this.init();
    }

    init() {
        // Create the job status container
        this.createContainer();

        // Start polling
        this.poll();
    }

    createContainer() {
        // Create floating job status indicator
        this.container = document.createElement('div');
        this.container.id = 'job-monitor';
        this.container.style.cssText = `
            position: fixed;
            bottom: 1rem;
            left: 1rem;
            z-index: 1000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 0.85rem;
        `;
        document.body.appendChild(this.container);
    }

    async poll() {
        try {
            const response = await fetch('/useradmin/api/jobs/active');
            const data = await response.json();

            const previousJobs = [...this.activeJobs];
            this.activeJobs = data.jobs || [];

            // Check for completed jobs (were active, now not)
            for (const prevJob of previousJobs) {
                const stillActive = this.activeJobs.find(j => j.id === prevJob.id);
                if (!stillActive) {
                    // Job completed - fetch final status
                    this.onJobComplete(prevJob.id);
                }
            }

            this.render();

        } catch (e) {
            console.error('Job monitor poll failed:', e);
        }

        // Schedule next poll
        const interval = this.activeJobs.length > 0 ? this.pollInterval : this.idleInterval;
        this.pollingTimer = setTimeout(() => this.poll(), interval);
    }

    async onJobComplete(jobId) {
        try {
            const response = await fetch(`/useradmin/api/jobs/${jobId}`);
            const job = await response.json();

            // Show completion notification
            this.showNotification(job);

            // Call any registered callbacks
            if (this.callbacks[jobId]) {
                this.callbacks[jobId](job);
                delete this.callbacks[jobId];
            }

            // Trigger custom event for page-specific handling
            window.dispatchEvent(new CustomEvent('jobComplete', { detail: job }));

        } catch (e) {
            console.error('Failed to get completed job status:', e);
        }
    }

    showNotification(job) {
        const notification = document.createElement('div');
        const isSuccess = job.status === 'completed';

        notification.style.cssText = `
            background: ${isSuccess ? '#4ecca3' : '#e94560'};
            color: ${isSuccess ? '#1a1a2e' : 'white'};
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease;
        `;

        const icon = isSuccess ? '[OK]' : '[X]';
        notification.innerHTML = `
            <strong>${icon} ${job.job_type}</strong>: ${job.source_id}<br>
            <span style="font-size: 0.8rem;">${job.message}</span>
        `;

        this.container.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transition = 'opacity 0.3s';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    render() {
        // Remove old job cards (keep notifications)
        const oldCards = this.container.querySelectorAll('.job-card');
        oldCards.forEach(card => card.remove());

        if (this.activeJobs.length === 0) {
            return;
        }

        // Add active job cards
        for (const job of this.activeJobs) {
            const card = document.createElement('div');
            card.className = 'job-card';
            card.style.cssText = `
                background: #16213e;
                border: 1px solid #0f3460;
                border-radius: 6px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                min-width: 250px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            `;

            const statusIcon = job.status === 'running' ? '...' : '[ ]';

            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong style="color: #e94560;">${statusIcon} ${job.job_type}</strong>
                    <span style="color: #4ecca3; font-size: 0.8rem;">${job.source_id}</span>
                </div>
                <div style="background: #0f3460; border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 0.5rem;">
                    <div style="background: #4ecca3; height: 100%; width: ${job.progress}%; transition: width 0.3s;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.75rem;">
                    <span>${job.message}</span>
                    <span>${job.progress}%</span>
                </div>
            `;

            this.container.appendChild(card);
        }
    }

    // Public API

    /**
     * Register a callback to be called when a specific job completes
     */
    onComplete(jobId, callback) {
        this.callbacks[jobId] = callback;
    }

    /**
     * Check if there's an active job for a source
     */
    hasActiveJob(sourceId) {
        return this.activeJobs.some(j => j.source_id === sourceId);
    }

    /**
     * Get active job for a source
     */
    getActiveJob(sourceId) {
        return this.activeJobs.find(j => j.source_id === sourceId);
    }

    /**
     * Manually trigger a poll (useful after submitting a job)
     */
    refresh() {
        if (this.pollingTimer) {
            clearTimeout(this.pollingTimer);
        }
        this.poll();
    }
}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// Create global instance
window.jobMonitor = new JobMonitor();
