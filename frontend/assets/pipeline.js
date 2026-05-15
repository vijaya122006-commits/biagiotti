/**
 * =============================================================================
 * pipeline.js — Biagiotti  |  Shared Pipeline State Manager
 * =============================================================================
 *
 * Manages product context across all pages using localStorage.
 * Every page that is part of the pipeline includes this script.
 *
 * Key functions:
 *   Pipeline.setProduct(id, name, ingredients)
 *   Pipeline.getProduct()        → { product_id, product_name, ingredients }
 *   Pipeline.clearProduct()
 *   Pipeline.setResult(key, obj) → store safety / forecast / similarity results
 *   Pipeline.getResult(key)
 *   Pipeline.clearResults()
 *   Pipeline.goTo(url)           → navigate with context preserved
 *   Pipeline.bannerHtml()        → returns context banner HTML string
 *   Pipeline.fetchProduct(id)    → GET /api/pipeline/product/:id
 *   Pipeline.renderBanner(containerId)
 *
 * Storage keys prefix: "biagiotti_"
 * =============================================================================
 */

(function (global) {
  'use strict';

  // ── Configuration ─────────────────────────────────────────────────────────
  var BASE_URL = window.__API_BASE_URL__ || 'http://127.0.0.1:5050';
  var PIPELINE_BASE = BASE_URL + '/api/pipeline';

  // ── Storage Keys ──────────────────────────────────────────────────────────
  var KEYS = {
    product: 'biagiotti_product',
    safetyResult: 'biagiotti_safety_result',
    forecastResult: 'biagiotti_forecast_result',
    similarResult: 'biagiotti_similar_result',
    searchQuery: 'biagiotti_search_query',
  };

  // ── Core Storage Helpers ──────────────────────────────────────────────────

  function _set(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) { /* quota */ }
  }

  function _get(key) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

  function _remove(key) {
    try { localStorage.removeItem(key); } catch (e) { }
  }

  // ── Product Context ───────────────────────────────────────────────────────

  function setProduct(id, name, ingredients, fullObj) {
    var p = fullObj || {
      product_id: String(id || ''),
      product_name: String(name || ''),
      ingredients: String(ingredients || ''),
    };
    _set(KEYS.product, p);
  }

  function getProduct() {
    return _get(KEYS.product) || null;
  }

  function clearProduct() {
    _remove(KEYS.product);
  }

  // ── Result Storage ────────────────────────────────────────────────────────

  function setResult(key, obj) {
    if (KEYS[key]) _set(KEYS[key], obj);
  }

  function getResult(key) {
    if (KEYS[key]) return _get(KEYS[key]);
    return null;
  }

  function clearResults() {
    _remove(KEYS.safetyResult);
    _remove(KEYS.forecastResult);
    _remove(KEYS.similarResult);
  }

  function clearAll() {
    clearProduct();
    clearResults();
    _remove(KEYS.searchQuery);
  }

  // ── Search Query (safety → similarity handoff) ────────────────────────────

  function setSearchQuery(q) { _set(KEYS.searchQuery, q); }
  function getSearchQuery() { return _get(KEYS.searchQuery) || null; }
  function clearSearchQuery() { _remove(KEYS.searchQuery); }

  // ── Navigation ────────────────────────────────────────────────────────────

  function goTo(url) {
    window.location.href = url;
  }

  function getHeaders() {
    return {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + localStorage.getItem('token')
    };
  }

  // ── API Fetch ─────────────────────────────────────────────────────────────

  async function fetchProduct(id) {
    var res = await fetch(PIPELINE_BASE + '/product/' + encodeURIComponent(id));
    var json = await res.json();
    if (!res.ok) throw new Error(json.message || 'Product fetch failed');
    return json.data;
  }

  async function fetchProducts(params) {
    var qs = new URLSearchParams(params || {}).toString();
    var res = await fetch(PIPELINE_BASE + '/products' + (qs ? '?' + qs : ''));
    var json = await res.json();
    if (!res.ok) throw new Error(json.message || 'Products fetch failed');
    return json.data;
  }

  async function fetchStats() {
    var res = await fetch(PIPELINE_BASE + '/stats');
    var json = await res.json();
    if (!res.ok) throw new Error(json.message || 'Stats fetch failed');
    return json.data;
  }

  async function fetchReport(id) {
    var res = await fetch(PIPELINE_BASE + '/report/' + encodeURIComponent(id));
    var json = await res.json();
    if (!res.ok) throw new Error(json.message || 'Report fetch failed');
    return json.data;
  }

  // ── Banner HTML ───────────────────────────────────────────────────────────

  function bannerHtml() {
    var p = getProduct();
    if (!p || !p.product_id) return '';
    return (
      '<div id="pipelineBanner" style="' +
      'background: linear-gradient(135deg, var(--primary-bg, #f9e8eb) 0%, #fdf2f4 100%);' +
      'border-left: 4px solid var(--primary, #b76e79);' +
      'border-radius: 8px;' +
      'padding: 0.75rem 1.25rem;' +
      'margin-bottom: 1.5rem;' +
      'display: flex;' +
      'align-items: center;' +
      'justify-content: space-between;' +
      'gap: 1rem;' +
      'font-size: 0.9rem;' +
      '">' +
      '<div style="display:flex; align-items:center; gap:0.75rem;">' +
      '<i class="fa-solid fa-link" style="color:var(--primary,#b76e79);"></i>' +
      '<span>' +
      '<strong style="color:var(--text-dark,#2c2a2a);">Pipeline: </strong>' +
      '<span style="color:var(--primary,#b76e79);">' + _escape(p.product_name || p.product_id) + '</span>' +
      ' <span style="color:var(--text-muted,#888); font-size:0.8rem;">(' + _escape(p.product_id) + ')</span>' +
      '</span>' +
      '</div>' +
      '<button onclick="Pipeline.clearAll(); location.reload();" ' +
      'style="background:none;border:none;cursor:pointer;color:var(--text-muted,#888);font-size:0.8rem;padding:0.2rem 0.5rem;">' +
      '<i class="fa-solid fa-xmark"></i> Clear' +
      '</button>' +
      '</div>'
    );
  }

  function renderBanner(containerId) {
    var el = document.getElementById(containerId);
    if (!el) return;
    var html = bannerHtml();
    if (html) el.insertAdjacentHTML('afterbegin', html);
  }

  function _escape(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // ── Toast Notification ────────────────────────────────────────────────────

  function toast(msg, type) {
    var t = document.createElement('div');
    var bg = type === 'error' ? '#c0392b'
      : type === 'warning' ? '#e67e22'
        : type === 'success' ? '#27ae60' : '#6c3eb5';
    t.style.cssText = (
      'position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;' +
      'background:' + bg + ';color:#fff;' +
      'padding:0.75rem 1.25rem;border-radius:8px;' +
      'font-size:0.9rem;box-shadow:0 4px 20px rgba(0,0,0,.18);' +
      'animation:slideInRight .3s ease;max-width:320px;'
    );
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(function () {
      t.style.opacity = '0';
      t.style.transition = 'opacity .3s';
      setTimeout(function () { t.remove(); }, 350);
    }, 3500);
  }

  // ── Inject keyframes if not already present ───────────────────────────────
  if (!document.getElementById('pipeline-keyframes')) {
    var style = document.createElement('style');
    style.id = 'pipeline-keyframes';
    style.textContent = '@keyframes slideInRight{from{transform:translateX(100%);opacity:0}to{transform:none;opacity:1}}';
    document.head.appendChild(style);
  }

  // ── Export ────────────────────────────────────────────────────────────────

  var Pipeline = {
    BASE: BASE_URL,
    PIPELINE_BASE: PIPELINE_BASE,
    // product context
    setProduct: setProduct,
    getProduct: getProduct,
    clearProduct: clearProduct,
    // results
    setResult: setResult,
    getResult: getResult,
    clearResults: clearResults,
    clearAll: clearAll,
    // search query handoff
    setSearchQuery: setSearchQuery,
    getSearchQuery: getSearchQuery,
    clearSearchQuery: clearSearchQuery,
    // navigation
    goTo: goTo,
    // api
    fetchProduct: fetchProduct,
    fetchProducts: fetchProducts,
    fetchStats: fetchStats,
    fetchReport: fetchReport,
    // ui
    bannerHtml: bannerHtml,
    renderBanner: renderBanner,
    toast: toast,
    getHeaders: getHeaders,
    API_BASE: BASE_URL,
  };

  global.Pipeline = Pipeline;

}(window));
