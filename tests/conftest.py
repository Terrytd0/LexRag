"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """FastAPI TestClient against the app, using startup/shutdown lifecycle."""
    with TestClient(app) as test_client:
        yield test_client
