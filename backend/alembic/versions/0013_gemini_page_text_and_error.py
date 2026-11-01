"""store Gemini page transcription, table layout and the processing failure reason

Revision ID: 0013_gemini_page_text_and_error
Revises: 0012_add_user_profiles

Adds nullable columns only. Existing rows keep working: pages without stored text read as empty.
Row-level security policies are per table, not per column, so they need no change.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0013_gemini_page_text_and_error"
down_revision = "0012_add_user_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("pages", recreate="always") as batch_op:
            batch_op.add_column(sa.Column("ocr_text", sa.Text(), nullable=True))
            batch_op.add_column(sa.Column("ocr_layout", sa.JSON(), nullable=True, server_default="[]"))
        with op.batch_alter_table("documents", recreate="always") as batch_op:
            batch_op.add_column(sa.Column("processing_error", sa.Text(), nullable=True))
        return

    op.add_column("pages", sa.Column("ocr_text", sa.Text(), nullable=True))
    op.add_column("pages", sa.Column("ocr_layout", sa.JSON(), nullable=True, server_default=sa.text("'[]'")))
    op.add_column("documents", sa.Column("processing_error", sa.Text(), nullable=True))


def downgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        with op.batch_alter_table("documents", recreate="always") as batch_op:
            batch_op.drop_column("processing_error")
        with op.batch_alter_table("pages", recreate="always") as batch_op:
            batch_op.drop_column("ocr_layout")
            batch_op.drop_column("ocr_text")
        return

    op.drop_column("documents", "processing_error")
    op.drop_column("pages", "ocr_layout")
    op.drop_column("pages", "ocr_text")
