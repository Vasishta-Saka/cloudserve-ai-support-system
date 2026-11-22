"""Pytest fixtures and configuration for CloudServe test suite."""

import pytest

from cloudserve.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    """Provide isolated application settings initialized with default unconfigured values."""
    return Settings(_env_file=None)
