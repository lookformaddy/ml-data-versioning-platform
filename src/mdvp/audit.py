"""Hash-chained, append-only audit log.

Each entry's hash covers the previous entry's hash plus this entry's
payload, so tampering, reordering, or deletion breaks the chain and
``verify()`` reports the first broken link. This is tamper-evidence,
not a signature scheme: a production deployment would sign entries
with a key-management service.
"""
from __future__ import annotations

from mdvp.hashing import canonical_json, sha256_hex
from mdvp.models import AuditEntry

GENESIS = "0" * 64


class AuditLog:
    """In-memory hash-chained audit log."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def append(
        self,
        timestamp: str,
        actor: str,
        action: str,
        dataset_id: str,
        version_id: str,
        detail: str,
    ) -> AuditEntry:
        prev_hash = self.entries[-1].entry_hash if self.entries else GENESIS
        seq = len(self.entries) + 1
        payload = canonical_json(
            {
                "seq": seq,
                "timestamp": timestamp,
                "actor": actor,
                "action": action,
                "dataset_id": dataset_id,
                "version_id": version_id,
                "detail": detail,
                "prev_hash": prev_hash,
            }
        )
        entry = AuditEntry(
            seq=seq,
            timestamp=timestamp,
            actor=actor,
            action=action,
            dataset_id=dataset_id,
            version_id=version_id,
            detail=detail,
            prev_hash=prev_hash,
            entry_hash=sha256_hex(payload),
        )
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, str]:
        """Replay the chain; return (valid, message)."""
        prev_hash = GENESIS
        for entry in self.entries:
            if entry.prev_hash != prev_hash:
                return False, f"broken link at seq {entry.seq}: prev_hash mismatch"
            payload = canonical_json(
                {
                    "seq": entry.seq,
                    "timestamp": entry.timestamp,
                    "actor": entry.actor,
                    "action": entry.action,
                    "dataset_id": entry.dataset_id,
                    "version_id": entry.version_id,
                    "detail": entry.detail,
                    "prev_hash": entry.prev_hash,
                }
            )
            if sha256_hex(payload) != entry.entry_hash:
                return False, f"tampered entry at seq {entry.seq}"
            prev_hash = entry.entry_hash
        return True, f"chain intact ({len(self.entries)} entries)"
