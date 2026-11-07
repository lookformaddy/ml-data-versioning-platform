"""Audit log tests: append, verify, tamper detection."""
from mdvp.audit import AuditLog
from tests.conftest import TODAY


def test_append_and_verify():
    log = AuditLog()
    log.append(TODAY, "a.chen", "version_committed", "churn", "1.0.0", "ok")
    log.append(TODAY, "d.kim", "promotion_applied", "churn", "1.0.0", "production")
    ok, message = log.verify()
    assert ok is True
    assert "2 entries" in message


def test_empty_log_verifies():
    ok, _ = AuditLog().verify()
    assert ok is True


def test_tamper_detected():
    log = AuditLog()
    log.append(TODAY, "a.chen", "version_committed", "churn", "1.0.0", "ok")
    log.entries[0].detail = "forged"
    ok, message = log.verify()
    assert ok is False
    assert "tampered" in message


def test_reorder_detected():
    log = AuditLog()
    log.append(TODAY, "a.chen", "a", "churn", "1.0.0", "1")
    log.append(TODAY, "a.chen", "b", "churn", "1.0.0", "2")
    log.entries.reverse()
    ok, message = log.verify()
    assert ok is False
