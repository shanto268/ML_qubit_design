from pathlib import Path

from squadds_ml_api.api import create_app


app = create_app(Path(__file__).resolve().parent)
