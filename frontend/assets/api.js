/**
 * assets/api.js — Centralized API calls for Cosmetic Intelligence
 */

const API_BASE = 'https://biagiotti-intelligence-jto5.onrender.com';

function _authHeaders() {
  const token = localStorage.getItem('token');
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };
}

async function _get(path, params = {}) {
  const url = new URL(API_BASE + path);
  Object.entries(params).forEach(([k, v]) => v !== undefined && url.searchParams.set(k, v));
  const res = await fetch(url.toString(), { headers: _authHeaders() });
  if (res.status === 401) { localStorage.clear(); window.location.href = 'index.html'; throw new Error('Unauthorized'); }
  return res.json();
}

async function _post(path, data = {}) {
  const res = await fetch(API_BASE + path, {
    method: 'POST', headers: _authHeaders(), body: JSON.stringify(data),
  });
  if (res.status === 401) { localStorage.clear(); window.location.href = 'index.html'; throw new Error('Unauthorized'); }
  return res.json();
}

async function _patch(path, data = {}) {
  const res = await fetch(API_BASE + path, {
    method: 'PATCH', headers: _authHeaders(), body: JSON.stringify(data),
  });
  return res.json();
}

const API = {
  // Auth
  login:    (data) => _post('/api/auth/login', data),
  register: (data) => _post('/api/auth/register', data),
  getMe:    ()     => _get('/api/auth/me'),

  // Dashboard
  getDashboardSummary: ()        => _get('/api/dashboard/summary'),
  getAlerts:           (unread)  => _get('/api/dashboard/alerts', { unread: unread ? 'true' : 'false' }),
  markAlertRead:       (id)      => _post(`/api/dashboard/alerts/${id}/read`),
  getPipelineStatus:   ()        => _get('/api/dashboard/pipeline-status'),
  refreshPipeline:     ()        => _post('/api/pipeline/refresh'),

  // Products
  getProducts:        (params)       => _get('/api/products', params),
  getProductAnalysis: (id)           => _get(`/api/products/${encodeURIComponent(id)}/analysis`),
  addProduct:         (data)         => _post('/api/products', data),
  updateStock:        (id, data)     => _patch(`/api/products/${encodeURIComponent(id)}/stock`, data),

  // Reviews
  getReviews:       (pid, params) => _get(`/api/products/${encodeURIComponent(pid)}/reviews`, params),
  addReview:        (pid, data)   => _post(`/api/products/${encodeURIComponent(pid)}/reviews`, data),
  getReviewSummary: (pid)         => _get(`/api/reviews/summary/${encodeURIComponent(pid)}`),

  // ML endpoints (direct)
  predictSkin:      (text)          => _post('/predict-skin', { text }),
  analyzeSentiment: (text)          => _post('/sentiment', { text }),
  detectHarmful:    (ingredients, name) => _post('/harmful', { ingredient_text: ingredients, product_name: name }),
  getSimilar:       (pid, top_n)    => _post('/similar-products', { product_id: pid, top_n: top_n || 5 }),
  forecastSales:    (features, steps, product_id, forecast_horizon) =>
    _post('/forecast', { features, steps, product_id, forecast_horizon }),

  // Health
  health: () => _get('/api/health'),
};

// Export for use in other scripts
if (typeof module !== 'undefined') module.exports = API;
