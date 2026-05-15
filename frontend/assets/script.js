/**
 * script.js — Biagiotti Global State & UX Manager
 * =============================================================================
 * Handles localStorage persistence, UI blocking if data is missing,
 * and global components like the "Data Loaded" banner.
 * =============================================================================
 */

(function (global) {
    'use strict';

    // ── Utilities ────────────────────────────────────────────────────────────

    function saveData(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
        } catch (e) {
            console.error('Error saving to localStorage', e);
        }
    }

    function getData(key) {
        try {
            const raw = localStorage.getItem(key);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    }

    function clearData() {
        localStorage.clear();
        window.location.href = 'upload.html';
    }

    // ── UI Components ────────────────────────────────────────────────────────

    function renderGlobalHeader() {
        const data = getData('salesData');
        if (!data) return;

        // Create banner if it doesn't exist
        if (document.getElementById('globalStateBanner')) return;

        const banner = document.createElement('div');
        banner.id = 'globalStateBanner';
        banner.style.cssText = `
            background: linear-gradient(90deg, #1a1a1a 0%, #2c2c2c 100%);
            color: #fff;
            padding: 8px 20px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10000;
            position: relative;
            border-bottom: 1px solid rgba(183, 110, 121, 0.3);
        `;

        const time = new Date(data.timestamp).toLocaleString();
        banner.innerHTML = `
            <div style="display:flex; align-items:center; gap:15px;">
                <span><i class="fa-solid fa-database" style="color:#b76e79;"></i> <strong>Active Dataset:</strong> ${data.filename || 'Unknown'}</span>
                <span><i class="fa-solid fa-layer-group" style="color:#b76e79;"></i> <strong>Items:</strong> ${data.uploaded_count || 0}</span>
                <span style="opacity:0.7;"><i class="fa-solid fa-clock"></i> Uploaded: ${time}</span>
            </div>
            <div style="display:flex; gap:10px;">
                <button id="globalResetBtn" style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); color:#fff; padding:3px 10px; border-radius:4px; cursor:pointer; font-size:11px; transition:all 0.2s;">
                    <i class="fa-solid fa-rotate"></i> Reset Data
                </button>
            </div>
        `;

        document.body.prepend(banner);

        document.getElementById('globalResetBtn').addEventListener('click', () => {
            if (confirm('Are you sure you want to clear all uploaded data and results?')) {
                clearData();
            }
        });
        
        // Hover effect for button
        const btn = document.getElementById('globalResetBtn');
        btn.onmouseover = () => { btn.style.background = 'rgba(183, 110, 121, 0.3)'; btn.style.borderColor = '#b76e79'; };
        btn.onmouseout = () => { btn.style.background = 'rgba(255,255,255,0.1)'; btn.style.borderColor = 'rgba(255,255,255,0.2)'; };
    }

    function blockUI() {
        // Don't block upload page or index
        const path = window.location.pathname;
        const hasToken = localStorage.getItem('token');
        
        // Don't block these pages
        if (path.includes('upload.html') || 
            path.includes('database_connection.html') || 
            path.includes('login.html') || 
            path === '/' || 
            path.endsWith('index.html')) return;

        // If we have an auth token, assume we might be using a Live DB connection
        if (hasToken) return;

        const data = getData('salesData');
        if (data) return;

        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(8px);
            z-index: 9999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 20px;
        `;

        overlay.innerHTML = `
            <div style="max-width: 400px; padding: 40px; background: #fff; border-radius: 15px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); border: 1px solid #f0f0f0;">
                <i class="fa-solid fa-cloud-arrow-up" style="font-size: 50px; color: #b76e79; margin-bottom: 20px;"></i>
                <h2 style="color: #2c2a2a; margin-bottom: 10px;">Pipeline Empty</h2>
                <p style="color: #7e7c7c; line-height: 1.6; margin-bottom: 25px;">Please upload your cosmetics sales data first to enable analysis and dashboard features.</p>
                <a href="upload.html" class="btn btn-primary" style="text-decoration: none; display: inline-block; padding: 12px 25px;">
                    Go to Upload Page
                </a>
            </div>
        `;

        document.body.appendChild(overlay);
        document.body.style.overflow = 'hidden';
    }

    // ── Initialization ───────────────────────────────────────────────────────

    function init() {
        renderGlobalHeader();
        blockUI();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // ── Export ───────────────────────────────────────────────────────────────

    global.State = {
        saveData: saveData,
        getData: getData,
        clearData: clearData,
        renderGlobalHeader: renderGlobalHeader
    };

})(window);
