"""
Operational health endpoints.
"""

import os
import time

from fastapi import APIRouter

router = APIRouter(tags=["operations"])

START_TIME = time.time()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "traffic-marl-backend",
        "timestamp": time.time(),
    }


@router.get("/ready")
def readiness():
    return {
        "status": "ready",
        "sumo_configured": bool(os.getenv("SUMO_HOME")),
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@router.get("/metrics")
def metrics():
    return {
        "service": "traffic-marl-backend",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "evaluation_metrics": [
            "reward",
            "waiting_time",
            "queue_length",
            "average_speed",
            "throughput",
        ],
    }