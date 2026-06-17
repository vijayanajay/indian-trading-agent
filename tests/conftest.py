import os
import pytest

# Force DB_PATH to use a test database file to isolate tests from production database
TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_trading_agent.db"))
os.environ["DB_PATH"] = TEST_DB_PATH

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    # Setup: remove existing test DB if any
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
            
    yield
    
    # Teardown: remove the test database file after all tests complete
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass
