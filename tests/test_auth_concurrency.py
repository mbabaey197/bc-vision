import sqlite3
import threading

from starlette.requests import Request

import app.main as main


def _request():
    return Request({
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/login",
        "raw_path": b"/login",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    })


def test_parallel_failed_logins_reliably_lock_account(tmp_path, monkeypatch):
    database = tmp_path / "parallel-login.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE users(
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                is_active INTEGER,
                failed_attempts INTEGER,
                locked_until TEXT,
                must_change_password INTEGER,
                session_version INTEGER,
                last_login TEXT
            );
            CREATE TABLE audit_logs(
                id INTEGER PRIMARY KEY,
                username TEXT,
                action TEXT,
                details TEXT,
                ip_address TEXT
            );
            CREATE TABLE revoked_sessions(
                token_hash TEXT PRIMARY KEY,
                expires_at INTEGER,
                revoked_at TEXT
            );
            INSERT INTO users(
                username,password_hash,is_active,failed_attempts,
                must_change_password,session_version
            ) VALUES('operator','hash',1,0,0,1);
        """)

    def connect():
        connection = sqlite3.connect(
            database,
            timeout=5,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection

    callers = 5
    verification_barrier = threading.Barrier(callers)

    def reject_password(_password, _password_hash):
        verification_barrier.wait(timeout=2)
        return False

    monkeypatch.setattr(main, "connect", connect)
    monkeypatch.setattr(main, "verify_password", reject_password)

    responses = []
    threads = [
        threading.Thread(
            target=lambda: responses.append(
                main.login(
                    _request(),
                    username="operator",
                    password="wrong",
                    next="/dashboard",
                )
            )
        )
        for _ in range(callers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(responses) == callers
    assert all(response.status_code == 303 for response in responses)
    with connect() as connection:
        user = connection.execute(
            "SELECT failed_attempts,locked_until FROM users "
            "WHERE username='operator'"
        ).fetchone()
        failures = connection.execute(
            "SELECT COUNT(*) FROM audit_logs WHERE action='login_failed'"
        ).fetchone()[0]
    assert user["failed_attempts"] == 0
    assert user["locked_until"]
    assert failures == callers
