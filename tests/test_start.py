import socket
import threading
import time
import pytest
from unittest.mock import MagicMock, patch
from start import check_port_listening, wait_for_backend, wait_for_frontend

def test_check_port_listening():
    # 1. Test closed port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    
    # It should return False when port is closed
    assert check_port_listening('127.0.0.1', port) is False

    # 2. Test open port
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 0))
    server_port = server_socket.getsockname()[1]
    server_socket.listen(1)
    
    def accept_one():
        try:
            conn, addr = server_socket.accept()
            conn.close()
        except Exception:
            pass

    t = threading.Thread(target=accept_one, daemon=True)
    t.start()
    
    try:
        # It should return True when port is open and listening
        assert check_port_listening('127.0.0.1', server_port) is True
    finally:
        server_socket.close()

@patch('urllib.request.urlopen')
def test_wait_for_backend_success(mock_urlopen):
    # Mock uvicorn process
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None # Process is running

    # Mock response 200 OK
    mock_response = MagicMock()
    mock_response.status = 200
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # Call should succeed immediately
    assert wait_for_backend("http://localhost:8000/api/health", mock_proc, timeout_seconds=5) is True
    assert mock_urlopen.call_count == 1

@patch('urllib.request.urlopen')
def test_wait_for_backend_timeout(mock_urlopen):
    # Mock uvicorn process
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None # Process is running

    # Mock response raising exception (timeout/connection error)
    mock_urlopen.side_effect = Exception("Connection refused")

    # Call should time out after timeout_seconds retries
    with patch('time.sleep') as mock_sleep:
        assert wait_for_backend("http://localhost:8000/api/health", mock_proc, timeout_seconds=3) is False
        assert mock_sleep.call_count == 3

def test_wait_for_backend_proc_died():
    # Mock uvicorn process that dies immediately
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1 # Process died

    # Call should fail immediately without retrying
    with patch('time.sleep') as mock_sleep:
        assert wait_for_backend("http://localhost:8000/api/health", mock_proc, timeout_seconds=5) is False
        assert mock_sleep.call_count == 0

@patch('start.check_port_listening')
def test_wait_for_frontend_success(mock_check_port):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    # Port is open
    mock_check_port.return_value = True

    assert wait_for_frontend("localhost", 3000, mock_proc, timeout_seconds=5) is True
    assert mock_check_port.call_count == 1

@patch('start.check_port_listening')
def test_wait_for_frontend_timeout(mock_check_port):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    # Port is closed
    mock_check_port.return_value = False

    with patch('time.sleep') as mock_sleep:
        assert wait_for_frontend("localhost", 3000, mock_proc, timeout_seconds=3) is False
        assert mock_check_port.call_count == 3
        assert mock_sleep.call_count == 3

def test_wait_for_frontend_proc_died():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1

    with patch('time.sleep') as mock_sleep:
        assert wait_for_frontend("localhost", 3000, mock_proc, timeout_seconds=5) is False
        assert mock_sleep.call_count == 0
