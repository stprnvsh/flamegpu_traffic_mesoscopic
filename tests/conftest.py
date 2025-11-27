"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "sumo: marks tests related to SUMO compatibility"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for floating point comparisons"""
    return {
        'rel': 0.01,  # 1% relative tolerance
        'abs': 1e-6   # Absolute tolerance for near-zero values
    }

