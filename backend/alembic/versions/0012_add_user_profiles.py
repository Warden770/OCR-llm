"""add application user profiles

Revision ID: 0012_add_user_profiles
Revises: 0011_add_attribute_page_spans
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0012_add_user_profiles"
down_revision = "0011_add_attribute_page_spans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("user_id", name="pk_user_profiles"),
    )

    if op.get_bind().dialect.name != "sqlite":
        op.execute("ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY")
        owner_context = "nullif(current_setting('app.user_id', true), '')"
        op.execute(
            "CREATE POLICY user_profiles_owner_policy ON user_profiles FOR ALL "
            f"USING (user_id = {owner_context}) WITH CHECK (user_id = {owner_context})"
        )


def downgrade() -> None:
    op.drop_table("user_profiles")
