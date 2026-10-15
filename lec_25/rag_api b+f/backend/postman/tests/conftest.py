import os

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture()
def client(tmp_path):
    settings = get_settings()
    settings.vector_db_path = str(tmp_path / 'vectors.sqlite3')
    settings.app_env = 'testing'
    settings.include_source_previews = False
    settings.min_similarity_score = 0.0
    with TestClient(app) as test_client:
        yield test_client
