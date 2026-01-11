"""
Database Module for Quantum Traffic Optimizer.

This module provides database connection management using SQLAlchemy
with async support for PostgreSQL.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


class DatabaseManager:
    """
    Database connection manager with async support.

    Provides connection pooling, session management, and health checks.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize database manager.

        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    @property
    def engine(self):
        """Get or create the database engine."""
        if self._engine is None and self.settings.DATABASE_ENABLED:
            self._engine = create_async_engine(
                self.settings.DATABASE_URL,
                echo=self.settings.DEBUG,
                pool_size=self.settings.DATABASE_POOL_SIZE,
                max_overflow=self.settings.DATABASE_MAX_OVERFLOW,
                pool_recycle=self.settings.DATABASE_POOL_RECYCLE,
                pool_pre_ping=True,  # Verify connections before use
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create the session factory."""
        if self._session_factory is None and self.engine is not None:
            self._session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._session_factory

    async def init_db(self) -> None:
        """
        Initialize database tables.

        Creates all tables defined in the models if they don't exist.
        """
        if not self.settings.DATABASE_ENABLED:
            logger.info("Database is disabled, skipping initialization")
            return

        if self._initialized:
            return

        try:
            async with self.engine.begin() as conn:
                # Import models to register them with Base
                from . import database_models  # noqa: F401

                await conn.run_sync(Base.metadata.create_all)
                self._initialized = True
                logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._initialized = False
            logger.info("Database connections closed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session as an async context manager.

        Yields:
            AsyncSession for database operations.

        Example:
            async with db_manager.session() as session:
                result = await session.execute(query)
        """
        if not self.settings.DATABASE_ENABLED:
            raise RuntimeError("Database is not enabled")

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if database is healthy, False otherwise.
        """
        if not self.settings.DATABASE_ENABLED:
            return True  # No database, so it's "healthy"

        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Yields:
        AsyncSession for database operations.

    Example:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    settings = get_settings()
    if not settings.DATABASE_ENABLED:
        yield None
        return

    async with db_manager.session() as session:
        yield session


async def init_database() -> None:
    """Initialize database on application startup."""
    await db_manager.init_db()


async def close_database() -> None:
    """Close database on application shutdown."""
    await db_manager.close()
