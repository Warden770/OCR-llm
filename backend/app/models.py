from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_documents_document_type", "document_type"),
        Index("idx_documents_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    extracted_attributes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    pages: Mapped[List["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    attributes: Mapped[List["Attribute"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="Attribute.document_id",
    )


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("document_id", "id", name="uq_pages_document_id_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_layout: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True, default=list)

    document: Mapped[Document] = relationship(back_populates="pages")
    attributes: Mapped[List["Attribute"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        foreign_keys="Attribute.page_id",
    )


class Attribute(Base):
    __tablename__ = "attributes"
    __table_args__ = (
        Index("idx_attributes_entity_type_label", "entity_type_label"),
        Index("idx_attributes_extracted_value", "extracted_value"),
        Index("idx_attributes_confidence_score", "confidence_score"),
        ForeignKeyConstraint(
            ["document_id", "page_id"],
            ["pages.document_id", "pages.id"],
            name="fk_attributes_page_document",
            ondelete="CASCADE",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    entity_type_label: Mapped[str] = mapped_column(String(128), nullable=False)
    extracted_value: Mapped[str] = mapped_column(Text, nullable=False)
    bounding_boxes: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    page_spans: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="model")

    page: Mapped[Page] = relationship(back_populates="attributes", foreign_keys=[page_id])
    document: Mapped[Document] = relationship(back_populates="attributes", foreign_keys=[document_id])
    corrections: Mapped[List["Correction"]] = relationship(back_populates="attribute", cascade="all, delete-orphan")


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    attribute_id: Mapped[int] = mapped_column(ForeignKey("attributes.id", ondelete="CASCADE"), nullable=False)
    original_prediction: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_value: Mapped[str] = mapped_column(Text, nullable=False)
    operator_user: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    attribute: Mapped[Attribute] = relationship(back_populates="corrections")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
