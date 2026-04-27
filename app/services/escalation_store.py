from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas import EscalationTicket


class EscalationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer_draft TEXT,
                    customer_id TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    human_response TEXT
                )
                """
            )

    def create(
        self,
        knowledge_base_id: str,
        question: str,
        reason: str,
        answer_draft: str | None,
        context: list[dict[str, Any]],
        customer_id: str | None = None,
        session_id: str | None = None,
    ) -> EscalationTicket:
        ticket_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        payload = (
            ticket_id,
            knowledge_base_id,
            question,
            reason,
            "open",
            answer_draft,
            customer_id,
            session_id,
            now,
            now,
            json.dumps(context),
            None,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO escalations (
                    id, knowledge_base_id, question, reason, status,
                    answer_draft, customer_id, session_id, created_at,
                    updated_at, context_json, human_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return self.get(ticket_id)

    def get(self, ticket_id: str) -> EscalationTicket:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM escalations WHERE id = ?",
                (ticket_id,),
            ).fetchone()
        if row is None:
            raise KeyError(ticket_id)
        return self._row_to_ticket(row)

    def list(self) -> list[EscalationTicket]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM escalations ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_ticket(row) for row in rows]

    def resolve(self, ticket_id: str, human_response: str) -> EscalationTicket:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE escalations
                SET status = ?, human_response = ?, updated_at = ?
                WHERE id = ?
                """,
                ("resolved", human_response, now, ticket_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(ticket_id)
        return self.get(ticket_id)

    def _row_to_ticket(self, row: sqlite3.Row) -> EscalationTicket:
        return EscalationTicket(
            id=row["id"],
            knowledge_base_id=row["knowledge_base_id"],
            question=row["question"],
            reason=row["reason"],
            status=row["status"],
            answer_draft=row["answer_draft"],
            customer_id=row["customer_id"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            context=json.loads(row["context_json"]),
            human_response=row["human_response"],
        )
