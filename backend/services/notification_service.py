"""
services/notification_service.py — Create and manage dealer notifications
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from database.models import db, Notification

logger = logging.getLogger("notification_service")


def create_notification(dealer_id: int, notif_type: str, title: str, message: str,
                         product_id: str = None, severity: str = 'medium'):
    """Create a new notification row."""
    try:
        # Deduplicate: check if an unread notification with same title exists for this product
        existing = Notification.query.filter_by(
            dealer_id=dealer_id, 
            product_id=product_id, 
            title=title, 
            is_read=False
        ).first()
        if existing:
            return existing

        notif = Notification(
            dealer_id=dealer_id,
            product_id=product_id,
            notif_type=notif_type,
            title=title,
            message=message,
            severity=severity,
            is_read=False,
        )
        db.session.add(notif)
        db.session.commit()
        return notif
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to create notification: %s", e)
        return None


def clear_old_notifications(dealer_id: int, days: int = 7):
    """Delete old read notifications older than `days` days."""
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        Notification.query.filter(
            Notification.dealer_id == dealer_id,
            Notification.is_read == True,
            Notification.created_at < cutoff
        ).delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to clear old notifications: %s", e)


def get_dealer_notifications(dealer_id: int, unread_only: bool = False) -> list:
    """Return all notifications for a dealer, sorted by severity and date."""
    q = Notification.query.filter_by(dealer_id=dealer_id)
    if unread_only:
        q = q.filter_by(is_read=False)

    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    notifs = q.order_by(Notification.created_at.desc()).all()
    notifs.sort(key=lambda n: (severity_order.get(n.severity, 4), -n.notif_id))

    return [{
        'id': n.notif_id,
        'type': n.notif_type,
        'title': n.title,
        'message': n.message,
        'product_id': n.product_id,
        'severity': n.severity,
        'is_read': n.is_read,
        'created_at': n.created_at.isoformat() + 'Z' if n.created_at else None,
    } for n in notifs]


def mark_notification_read(notif_id: int, dealer_id: int) -> bool:
    """Mark a notification as read. Returns True on success."""
    try:
        n = Notification.query.filter_by(notif_id=notif_id, dealer_id=dealer_id).first()
        if not n:
            return False
        n.is_read = True
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to mark notification read: %s", e)
        return False
