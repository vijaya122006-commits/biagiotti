// BIAGIOTTI Theme - interactions and charts
// Updated to support dynamic data and global state management

(function (global) {
    'use strict';

    // ── Primary Biagiotti Colors for Charts ─────────────────────────────────────
    const chartColors = {
        primary: 'rgba(183, 110, 121, 0.8)', // Rose Gold
        primaryLight: 'rgba(183, 110, 121, 0.2)',
        safe: 'rgba(39, 174, 96, 0.8)',
        danger: 'rgba(211, 84, 0, 0.8)',
        warning: 'rgba(230, 126, 34, 0.8)',
        neutral: 'rgba(236, 240, 241, 0.8)',
        palette: [
            'rgba(183, 110, 121, 0.8)',
            'rgba(240, 187, 195, 0.8)',
            'rgba(215, 149, 161, 0.8)',
            'rgba(248, 195, 205, 0.8)',
            'rgba(161, 93, 103, 0.8)'
        ]
    };

    // ── Initialization ──────────────────────────────────────────────────────────

    function initGlobalUI() {
        // 1. Navbar Active State
        const navLinks = document.querySelectorAll('.nav-links a');
        const currentUrl = window.location.pathname.split('/').pop() || 'index.html';
        
        navLinks.forEach(link => {
            const linkUrl = link.getAttribute('href');
            if (linkUrl === currentUrl || (currentUrl === 'index.html' && linkUrl === 'dashboard.html')) {
                link.classList.add('active');
            }
        });

        // 2. Chart.js Defaults
        if (typeof Chart !== 'undefined') {
            Chart.defaults.font.family = "'Poppins', sans-serif";
            Chart.defaults.color = '#7E7C7C';
            Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(44, 42, 42, 0.9)';
            Chart.defaults.plugins.tooltip.padding = 10;
            Chart.defaults.plugins.tooltip.cornerRadius = 8;
            Chart.defaults.plugins.tooltip.displayColors = false;
        }

        // 3. Shared Upload UI Logic (used in upload.html and dashboard)
        const uploadArea = document.getElementById('dragDropArea');
        if(uploadArea) {
            uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('active'); });
            uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('active'); });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('active');
                const fileText = document.getElementById('fileName');
                if(e.dataTransfer.files.length > 0) {
                    fileText.textContent = `File selected: ${e.dataTransfer.files[0].name}`;
                }
            });
            uploadArea.addEventListener('click', () => {
                 const fileInput = document.getElementById('fileInputReal');
                 if(fileInput) fileInput.click();
            });
        }
    }

    // ── Chart Factory ───────────────────────────────────────────────────────────

    function initDashSalesChart(canvasId, labels, data) {
        const ctx = document.getElementById(canvasId);
        if(!ctx) return;
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels || ['N/A'],
                datasets: [{
                    label: 'Sales ($)',
                    data: data || [0],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primaryLight,
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { borderDash: [5, 5] } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    function initDashCategoryChart(canvasId, labels, data) {
        const ctx = document.getElementById(canvasId);
        if(!ctx) return;
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels || ['No Data'],
                datasets: [{
                    data: data || [100],
                    backgroundColor: data ? chartColors.palette : [chartColors.neutral],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    function initForecastChart(canvasId, labels, data) {
        const ctx = document.getElementById(canvasId);
        if(!ctx) return;
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels || ['Day 1', 'Day 7', 'Day 14', 'Day 21', 'Day 30'],
                datasets: [{
                    label: 'Forecasted Demand',
                    data: data || [0, 0, 0, 0, 0],
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primaryLight,
                    borderWidth: 3,
                    tension: 0.4,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: chartColors.primary,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });
    }

    function initSkinSentimentChart(canvasId, labels, data) {
        const ctx = document.getElementById(canvasId);
        if(!ctx) return;
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels || ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: data || [0, 0, 0],
                    backgroundColor: [chartColors.safe, chartColors.neutral, chartColors.danger],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // ── Export ──────────────────────────────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', initGlobalUI);

    global.BiagiottiUI = {
        initDashSalesChart:     initDashSalesChart,
        initDashCategoryChart:  initDashCategoryChart,
        initForecastChart:      initForecastChart,
        initSkinSentimentChart: initSkinSentimentChart,
        colors:                 chartColors
    };

})(window);
