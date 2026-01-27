"""
Pytest configuration for V-Gate tests.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']
