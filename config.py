"""
Configuration settings for the CRM-Agent system.
This module handles loading environment variables, API keys, and other settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
DATA_DIR = OUTPUT_DIR / "data"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, REPORTS_DIR, VISUALIZATIONS_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Google Cloud settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "asia-northeast3")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# BigQuery settings
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "PSEUDO_CDP")
BQ_CUSTOMER_TABLE = os.getenv("BQ_CUSTOMER_TABLE", "customer_master")
BQ_PRODUCT_TABLE = os.getenv("BQ_PRODUCT_TABLE", "product_master")
BQ_OFFLINE_TRANSACTIONS_TABLE = os.getenv("BQ_OFFLINE_TRANSACTIONS_TABLE", "offline_transactions")
BQ_ONLINE_BEHAVIOR_TABLE = os.getenv("BQ_ONLINE_BEHAVIOR_TABLE", "online_behavior")

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))

# Agent settings
TREND_ANALYSIS_TIMEFRAME_DAYS = int(os.getenv("TREND_ANALYSIS_TIMEFRAME_DAYS", "365"))
FORECAST_HORIZON_WEEKS = int(os.getenv("FORECAST_HORIZON_WEEKS", "12"))
DEFAULT_NUM_CAMPAIGNS = int(os.getenv("DEFAULT_NUM_CAMPAIGNS", "5"))
DISCUSSION_MAX_ITERATIONS = int(os.getenv("DISCUSSION_MAX_ITERATIONS", "3"))

# Web app settings
APP_TITLE = "가전 리테일 CDP AI Agent 시스템"
APP_DESCRIPTION = "CDP 데이터를 활용한 트렌드 분석, 수요 예측, 캠페인 기획 및 세그먼트 추출 시스템"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Sample .env file content (for reference)
SAMPLE_ENV = """
# Google Cloud settings
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=asia-northeast3
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

# BigQuery settings
BQ_DATASET_ID=PSEUDO_CDP
BQ_CUSTOMER_TABLE=customer_master
BQ_PRODUCT_TABLE=product_master
BQ_OFFLINE_TRANSACTIONS_TABLE=offline_transactions
BQ_ONLINE_BEHAVIOR_TABLE=online_behavior

# Gemini API settings
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-1.5-pro
GEMINI_TEMPERATURE=0.2

# Agent settings
TREND_ANALYSIS_TIMEFRAME_DAYS=365
FORECAST_HORIZON_WEEKS=12
DEFAULT_NUM_CAMPAIGNS=5
DISCUSSION_MAX_ITERATIONS=3

# Web app settings
DEBUG_MODE=False
"""

def get_env_status():
    """Check if all required environment variables are set"""
    required_vars = ["GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        return {
            "status": "missing",
            "missing_vars": missing_vars
        }
    return {"status": "ok"}

def print_config_summary():
    """Print a summary of the configuration"""
    print("=== CRM-Agent Configuration ===")
    print(f"GCP Project: {GCP_PROJECT_ID}")
    print(f"BigQuery Dataset: {BQ_DATASET_ID}")
    print(f"Gemini Model: {GEMINI_MODEL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Debug Mode: {DEBUG_MODE}")
    
    env_status = get_env_status()
    if env_status["status"] == "missing":
        print("\nWARNING: Missing required environment variables:")
        for var in env_status["missing_vars"]:
            print(f"  - {var}")
        print("\nPlease create a .env file with the required variables.")
    else:
        print("\nAll required environment variables are set.")

if __name__ == "__main__":
    print_config_summary()
