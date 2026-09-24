from fastapi import FastAPI, Depends
from loguru import logger
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables (useful for local testing outside Docker)
load_dotenv()

# Configure Loguru Logging
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
logger.add("logs/application.log", rotation="10 MB", level="DEBUG")

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/traffic_ai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Traffic MARL API", version="0.1.0")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Traffic MARL API...")
    # Basic connection test
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Traffic AI Running"}

@app.get("/health")
def health_check(db=Depends(get_db)):
    logger.info("Health check endpoint accessed")
    db_status = "Connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "Disconnected"
        
    return {
        "server": "Running",
        "database": db_status,
        "version": app.version
    }

@app.get("/version")
def get_version():
    return {"version": app.version}