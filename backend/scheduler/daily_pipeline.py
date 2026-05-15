"""
scheduler/daily_pipeline.py — APScheduler daily pipeline at 2 AM
"""
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger("daily_pipeline")


def run_all_dealer_pipelines(app):
    """Called every day at 2 AM. Refreshes analysis for all active dealers."""
    from database.models import Dealer
    from services.analysis_engine import run_full_pipeline_for_dealer

    with app.app_context():
        dealers = Dealer.query.all()
        for dealer in dealers:
            try:
                logger.info("Running daily pipeline for dealer %d", dealer.dealer_id)
                run_full_pipeline_for_dealer(dealer.dealer_id)
                logger.info("Pipeline complete for dealer %d", dealer.dealer_id)
            except Exception as e:
                logger.error("Pipeline failed for dealer %d: %s", dealer.dealer_id, e)


def init_scheduler(app):
    """Initialize APScheduler and schedule the daily pipeline."""
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=run_all_dealer_pipelines,
        args=(app,),
        trigger='cron',
        hour=2,
        minute=0,
        id='daily_pipeline',
        replace_existing=True,
        max_instances=1,
    )

    try:
        scheduler.start()
        logger.info("Scheduler started — daily pipeline runs at 2 AM")
    except Exception as e:
        logger.error("Failed to start scheduler: %s", e)

    return scheduler
