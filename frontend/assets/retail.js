document.addEventListener('DOMContentLoaded', async () => {
    try {
        await initRetailEngine();
    } catch (e) {
        console.error("Retail Dashboard Init Failed:", e);
    }
});

let globalRetailData = [];
let modalChartInstance = null;

async function initRetailEngine() {
    // 1. Fetch products from Pipeline Backend
    const pipelineData = await Pipeline.fetchProducts({ limit: 12 });
    const products = pipelineData.products || [];
    
    // 2. Ask ML Engine for Business Forecast & Inventory Status
    const promises = products.map(async (p) => {
        try {
            // Sending [100] as a bootstrap triggers the ML service to synthesize
            // 2 years of product-specific history using the product hash logic
            const res = await window.API.forecastSales([100], 8, p.product_id);
            res.product_name = p.product_name; // inject name for UI
            return res;
        } catch (e) {
            console.warn("Skipping product due to API error", e);
            return null;
        }
    });

    const results = await Promise.all(promises);
    globalRetailData = results.filter(r => r !== null && r.status === 'success');
    
    // 3. Feed the UI Layout
    updateKPIs(globalRetailData);
    renderPriorityAlerts(globalRetailData);
    renderProductGrid(globalRetailData);
}

// ── KPI Logic ──
function updateKPIs(data) {
    document.getElementById('kpiTotal').textContent = data.length;
    
    const riskCount = data.filter(d => d.stockout_risk === 'HIGH').length;
    document.getElementById('kpiRisk').textContent = riskCount;
    
    const growCount = data.filter(d => d.demand_change_pct > 2).length;
    document.getElementById('kpiGrow').textContent = growCount;
}

// ── The Decision Engine Color Mapping ──
function getDecisionClass(decisionStr) {
    const d = decisionStr.toUpperCase();
    if (d.includes('AGGRESSIVE') || d.includes('CLEARANCE')) return 'danger';
    if (d.includes('CAREFULLY') || d.includes('CLOSELY') || d.includes('REDUCE')) return 'warning';
    if (d.includes('INCREASE')) return 'success';
    return 'info'; // MAINTAIN
}

function getDecisionBadge(decisionStr) {
    const cls = getDecisionClass(decisionStr);
    const short = decisionStr.split(' | ')[0]; // remove tie breakers if any
    return `<span class="decision-badge badge-${cls}">${short}</span>`;
}

// ── Alert Panel ──
function renderPriorityAlerts(data) {
    const container = document.getElementById('alertsContainer');
    
    // Sort descending by priority_score
    const sorted = [...data].sort((a, b) => b.priority_score - a.priority_score);
    const topAlerts = sorted.slice(0, 3);
    
    if (topAlerts.length === 0) {
        container.innerHTML = `<div class="alert-item"><div class="alert-main"><span class="alert-product text-muted">All systems nominal</span></div></div>`;
        return;
    }

    container.innerHTML = topAlerts.map(r => {
        const dClass = getDecisionClass(r.decision);
        let alertUrgency = 'good';
        if (dClass === 'danger') alertUrgency = 'urgent';
        else if (dClass === 'warning') alertUrgency = 'warning';

        return `
            <div class="alert-item ${alertUrgency}">
                <div class="alert-main">
                    <span class="alert-product">${r.product_name}</span>
                    <span class="alert-reason">${r.explanation}</span>
                </div>
                <div class="alert-action ${'text-' + dClass}">
                    ${r.decision.split(' | ')[0]}
                </div>
            </div>
        `;
    }).join('');
}

// ── Grid Mapping ──
function renderProductGrid(data) {
    const container = document.getElementById('productGrid');
    container.innerHTML = data.map((r, i) => `
        <div class="product-card" onclick="openModal(${i})">
            <div class="card-header">
                <div class="card-title">${r.product_name}</div>
                ${getDecisionBadge(r.decision)}
            </div>
            
            <!-- Mini Chart -->
            <div class="chart-container-mini">
                <canvas id="miniChart_${i}"></canvas>
            </div>
            
            <!-- Metrics -->
            <div class="card-metrics">
                <div class="metric">
                    <span class="metric-label">Inv. Left</span>
                    <span class="metric-value">${r.days_of_inventory} Days</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Demand</span>
                    <span class="metric-value ${r.demand_change_pct > 0 ? 'text-success' : 'text-danger'}">
                        ${r.demand_change_pct > 0 ? '+' : ''}${r.demand_change_pct.toFixed(1)}%
                    </span>
                </div>
            </div>
        </div>
    `).join('');

    // Draw Mini Charts
    data.forEach((r, i) => drawChart(`miniChart_${i}`, r.history, r.forecast, true));
}

// ── Charts Implementation (Chart.js) ──
function drawChart(canvasId, history, forecast, isMini = false) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const fullData = [...history, ...forecast];
    const labels = fullData.map((_, idx) => `W${idx+1}`);
    
    const histPoints = history.length;
    const pastData = [...history, ...Array(forecast.length).fill(null)];
    const futureData = Array(histPoints - 1).fill(null);
    futureData.push(history[history.length - 1]); // connect points
    futureData.push(...forecast);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    data: pastData,
                    borderColor: '#64748b',
                    borderWidth: isMini ? 2 : 3,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    data: futureData,
                    borderColor: '#3b82f6',
                    borderWidth: isMini ? 2 : 3,
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: isMini ? 0 : 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: !isMini } },
            scales: {
                x: { display: !isMini },
                y: { display: !isMini, beginAtZero: false, border: {display: false} }
            },
            interaction: {
                intersect: false,
                mode: 'index',
            },
        }
    });
}

// ── Modal Interaction ──
window.openModal = function(index) {
    const r = globalRetailData[index];
    if (!r) return;
    
    // Text population
    document.getElementById('modalTitle').textContent = r.product_name;
    document.getElementById('modalExplanation').textContent = r.explanation;
    
    document.getElementById('modalDays').textContent = `${r.days_of_inventory} Days`;
    document.getElementById('modalReorder').textContent = `${r.reorder_point} Units`;
    document.getElementById('modalMargin').textContent = `${(r.profit_margin * 100).toFixed(1)}%`;
    
    const dem = document.getElementById('modalDemand');
    dem.textContent = `${r.demand_change_pct > 0 ? '+' : ''}${r.demand_change_pct.toFixed(1)}%`;
    dem.className = `metric-value ${r.demand_change_pct > 0 ? 'text-success' : 'text-danger'}`;
    
    // Badges
    const dClass = getDecisionClass(r.decision);
    const badge = document.getElementById('modalDecisionBadge');
    badge.className = `decision-badge badge-${dClass}`;
    badge.textContent = r.decision.split(' | ')[0];
    
    const risk = document.getElementById('modalRiskBadge');
    risk.textContent = `RISK: ${r.stockout_risk}`;
    risk.className = r.stockout_risk === 'HIGH' ? 'text-danger' : 
                     (r.stockout_risk === 'MEDIUM' ? 'text-warning' : 'text-info');
    
    // Large Chart
    if (modalChartInstance) {
        modalChartInstance.destroy();
    }
    document.getElementById('detailModal').classList.add('active');
    
    // Draw via slightly modified logic to expose instance
    setTimeout(() => {
        const ctx = document.getElementById('modalChart').getContext('2d');
        const pastData = [...r.history, ...Array(r.forecast.length).fill(null)];
        const futureData = Array(r.history.length - 1).fill(null);
        futureData.push(r.history[r.history.length - 1]);
        futureData.push(...r.forecast);

        modalChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [...r.history, ...r.forecast].map((_, i) => `W${i+1}`),
                datasets: [
                    {
                        label: 'Historical',
                        data: pastData,
                        borderColor: '#64748b',
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        backgroundColor: 'rgba(100, 116, 139, 0.05)'
                    },
                    {
                        label: 'Forecast',
                        data: futureData,
                        borderColor: '#3b82f6',
                        borderWidth: 3,
                        borderDash: [5, 5],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } }
            }
        });
    }, 50);
}

window.closeModal = function() {
    document.getElementById('detailModal').classList.remove('active');
}
