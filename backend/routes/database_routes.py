"""
routes/database_routes.py
──────────────────────────
REST endpoints for the Live Database Connection feature.

  POST /api/db-connect/test     — test a connection without saving
  POST /api/db-connect/save     — save credentials + start auto-sync
  GET  /api/db-connect/status   — get current sync status
  POST /api/db-connect/sync     — manually trigger a sync right now
  DELETE /api/db-connect        — remove the connection + stop sync
"""
import json
import logging
import threading

from flask import Blueprint, request, jsonify
from utils.auth import require_auth
from services.database_sync_service import (
    test_connection,
    sync_dealer_database,
    start_auto_sync,
    stop_auto_sync,
    _encode_password,
)

db_connect_bp = Blueprint('db_connect', __name__)
logger = logging.getLogger("database_routes")


# ── POST /api/db-connect/test ──────────────────────────────────────────────────
@db_connect_bp.route('/test', methods=['POST'])
@require_auth
def test_db_connection():
    """Test a live connection to the dealer's external database."""
    data = request.get_json(silent=True) or {}

    required = ['db_type', 'db_name']
    if data.get('db_type', '').lower() != 'sqlite':
        required += ['host', 'port', 'username', 'password']
    missing = [f for f in required if not str(data.get(f, '')).strip()]
    if missing:
        return jsonify({'success': False, 'message': f'Missing fields: {", ".join(missing)}'}), 400

    result = test_connection(
        db_type  = data['db_type'],
        host     = data.get('host', ''),
        port     = int(data.get('port', 0) or 0),
        db_name  = data['db_name'],
        username = data.get('username', ''),
        password = data.get('password', ''),
    )
    status = 200 if result['success'] else 400
    return jsonify(result), status


# ── POST /api/db-connect/save ──────────────────────────────────────────────────
@db_connect_bp.route('/save', methods=['POST'])
@require_auth
def save_db_connection():
    """
    Save the dealer's external database credentials and start auto-sync.
    Body fields:
      db_type, host, port, db_name, username, password   (required)
      products_table, sales_table                         (optional, default: products/sales)
      column_map    { products: {biagiotti_col: dealer_col}, sales: {...} }  (optional)
      sync_interval_min                                   (optional, default: 30)
    """
    from database.models import db, DealerDatabaseConnection
    from flask import current_app

    dealer_id = request.dealer_id
    data = request.get_json(silent=True) or {}

    required = ['db_type', 'db_name']
    if data.get('db_type', '').lower() != 'sqlite':
        required += ['host', 'port', 'username', 'password']
    missing = [f for f in required if not str(data.get(f, '')).strip()]
    if missing:
        return jsonify({'success': False, 'message': f'Missing fields: {", ".join(missing)}'}), 400

    # Test the connection first — refuse to save a broken one
    test_result = test_connection(
        db_type  = data['db_type'],
        host     = data['host'],
        port     = int(data['port']),
        db_name  = data['db_name'],
        username = data['username'],
        password = data['password'],
    )
    if not test_result['success']:
        return jsonify({'success': False, 'message': f"Connection test failed: {test_result['message']}"}), 400

    try:
        # Upsert connection record
        conn_row = DealerDatabaseConnection.query.filter_by(dealer_id=dealer_id).first()
        if not conn_row:
            conn_row = DealerDatabaseConnection(dealer_id=dealer_id)
            db.session.add(conn_row)

        conn_row.db_type            = data['db_type'].lower()
        conn_row.host               = data['host']
        conn_row.port               = int(data['port'])
        conn_row.db_name            = data['db_name']
        conn_row.username           = data['username']
        conn_row.password_encrypted = _encode_password(data['password'])
        conn_row.products_table     = data.get('products_table', 'products')
        conn_row.sales_table        = data.get('sales_table', 'sales')
        conn_row.column_map_json    = json.dumps(data.get('column_map', {}))
        conn_row.sync_interval_min  = int(data.get('sync_interval_min', 30))
        conn_row.is_active          = True
        conn_row.sync_status        = 'idle'

        db.session.commit()

        # Kick off the first sync immediately in background
        app = current_app._get_current_object()
        threading.Thread(
            target=_initial_sync,
            args=(dealer_id, app),
            daemon=True,
        ).start()

        return jsonify({
            'success': True,
            'message': 'Database connected! First sync starting in background.',
            'tables_found': test_result.get('tables', []),
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error("Save DB connection failed for dealer %d: %s", dealer_id, e)
        return jsonify({'success': False, 'message': str(e)}), 500


def _initial_sync(dealer_id: int, app):
    """Run first sync then start auto-sync loop."""
    with app.app_context():
        result = sync_dealer_database(dealer_id)
        logger.info("Initial sync for dealer %d: %s", dealer_id, result)
        if result.get('success'):
            # Run the ML pipeline after data is loaded
            try:
                from services.analysis_engine import run_full_pipeline_for_dealer
                run_full_pipeline_for_dealer(dealer_id)
            except Exception as e:
                logger.error("Pipeline after initial sync failed: %s", e)
    # Start the recurring auto-sync loop
    start_auto_sync(dealer_id, app)


# ── GET /api/db-connect/status ─────────────────────────────────────────────────
@db_connect_bp.route('/status', methods=['GET'])
@require_auth
def get_db_status():
    """Return the current sync status for this dealer's database connection."""
    from database.models import DealerDatabaseConnection
    dealer_id = request.dealer_id

    conn_row = DealerDatabaseConnection.query.filter_by(dealer_id=dealer_id).first()
    if not conn_row:
        return jsonify({'connected': False, 'message': 'No database connection configured.'}), 200

    return jsonify({
        'connected':        True,
        'db_type':          conn_row.db_type,
        'host':             conn_row.host,
        'port':             conn_row.port,
        'db_name':          conn_row.db_name,
        'products_table':   conn_row.products_table,
        'sales_table':      conn_row.sales_table,
        'sync_interval_min': conn_row.sync_interval_min,
        'is_active':        conn_row.is_active,
        'sync_status':      conn_row.sync_status,
        'last_synced_at':   conn_row.last_synced_at.isoformat() + 'Z' if conn_row.last_synced_at else None,
        'last_sync_rows':   conn_row.last_sync_rows,
        'last_error':       conn_row.last_error,
    }), 200


# ── POST /api/db-connect/sync ──────────────────────────────────────────────────
@db_connect_bp.route('/sync', methods=['POST'])
@require_auth
def manual_sync():
    """Manually trigger a sync right now (don't wait for the scheduled interval)."""
    from flask import current_app
    dealer_id = request.dealer_id
    app = current_app._get_current_object()

    threading.Thread(
        target=_run_sync_and_pipeline,
        args=(dealer_id, app),
        daemon=True,
    ).start()

    return jsonify({'success': True, 'message': 'Sync started. Check /status for progress.'}), 202


def _run_sync_and_pipeline(dealer_id: int, app):
    with app.app_context():
        result = sync_dealer_database(dealer_id)
        if result.get('success'):
            try:
                from services.analysis_engine import run_full_pipeline_for_dealer
                run_full_pipeline_for_dealer(dealer_id)
            except Exception as e:
                logger.error("Pipeline after manual sync failed: %s", e)


# ── DELETE /api/db-connect ─────────────────────────────────────────────────────
@db_connect_bp.route('', methods=['DELETE'])
@require_auth
def disconnect_database():
    """Remove the dealer's external database connection and stop auto-sync."""
    from database.models import db, DealerDatabaseConnection
    dealer_id = request.dealer_id

    conn_row = DealerDatabaseConnection.query.filter_by(dealer_id=dealer_id).first()
    if not conn_row:
        return jsonify({'success': False, 'message': 'No connection to remove.'}), 404

    stop_auto_sync(dealer_id)
    db.session.delete(conn_row)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Database disconnected.'}), 200
