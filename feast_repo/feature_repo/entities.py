# feature_repo/entities.py
from feast import Entity, ValueType

stock = Entity(
    name="stock",
    join_keys=["symbol"],
    value_type=ValueType.STRING,
    description="NSE stock symbol (e.g., TCS, INFY).",
)
