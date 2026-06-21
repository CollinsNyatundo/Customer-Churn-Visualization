"""
feature_store/features/entities.py
------------------------------------
Feast entity definitions.

An Entity is the primary key that all FeatureViews share.
Here: customer_id (string) ties together BCG, CRM, Support, and Billing views.
"""

from feast import Entity, ValueType

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=ValueType.STRING,
    description="Unique customer identifier across all data sources",
)
