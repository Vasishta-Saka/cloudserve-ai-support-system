"""Database engine and session management for CloudServe SQLite storage.

Configures:
- SQLAlchemy 2.0 engine creation with SQLite optimizations (WAL, NORMAL synchronous,
  foreign keys, busy timeout).
- Thread-safe connection pool handling for file-based and in-memory SQLite databases.
- Schema creation (`init_db`) via `Base.metadata.create_all`.
- Session factory and context management.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cloudserve.storage.models import Base

logger = logging.getLogger(__name__)


def _ensure_sqlite_parent_dir(url: str) -> None:
    """Ensure parent directory exists for file-based SQLite database URLs."""
    if not url.startswith("sqlite:///"):
        return
    file_path = url[len("sqlite:///") :]
    if file_path in (":memory:", "", "?cache=shared"):
        return
    # Strip any query parameters
    raw_path = file_path.split("?")[0]
    try:
        p = Path(raw_path)
        if not p.is_absolute():
            p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning(
            "Could not ensure directory for SQLite path '%s': %s",
            raw_path,
            exc,
        )


def _configure_sqlite_pragmas(engine: Engine, timeout_seconds: float = 5.0) -> None:
    """Register connect event listener setting standard SQLite pragmas."""

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)};")

            # Check if this is a file-backed database (WAL is only for disk databases)
            cursor.execute("PRAGMA database_list;")
            rows = cursor.fetchall()
            is_file_db = False
            for row in rows:
                if len(row) >= 3 and row[1] == "main" and row[2]:
                    is_file_db = True
                    break

            if is_file_db:
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA synchronous = NORMAL;")
        except Exception as exc:
            logger.debug("Failed to set SQLite pragma: %s", exc)
        finally:
            cursor.close()


def create_sqlite_engine(
    url: str,
    echo: bool = False,
    timeout_seconds: float = 5.0,
) -> Engine:
    """Create a configured SQLAlchemy engine for SQLite.

    Handles in-memory vs disk-based configuration:
    - Disk databases: sets parent directory, WAL journal mode, NORMAL sync, busy timeout.
    - In-memory databases: uses StaticPool and check_same_thread=False so sessions share state.
    """
    is_memory = (
        url in ("sqlite://", "sqlite:///:memory:")
        or ":memory:" in url
        or "mode=memory" in url
    )

    if is_memory:
        engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        _ensure_sqlite_parent_dir(url)
        engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False, "timeout": timeout_seconds},
        )

    _configure_sqlite_pragmas(engine, timeout_seconds=timeout_seconds)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a thread-safe SQLAlchemy sessionmaker for the given engine."""
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=True)


def init_db(engine: Engine) -> None:
    """Create all CloudServe database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        res = conn.exec_driver_sql("PRAGMA table_info(decision_audit_logs);").fetchall()
        if res:
            cols = {row[1] for row in res}
            if "processing_run_id" not in cols:
                conn.exec_driver_sql(
                    "ALTER TABLE decision_audit_logs ADD COLUMN processing_run_id VARCHAR(36);"
                )
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_audit_run_id ON "
                    "decision_audit_logs (processing_run_id);"
                )
    logger.info("Initialized CloudServe database schema.")


@contextmanager
def get_db_session(
    session_factory: sessionmaker[Session],
) -> Generator[Session, None, None]:
    """Context manager providing a transactional database session."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
