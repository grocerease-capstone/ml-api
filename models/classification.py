from typing import List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    product_names: List[str]


class PredictedClassificationItem(BaseModel):
    product_name: str
    category: str
