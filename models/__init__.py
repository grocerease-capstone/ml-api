from typing import List
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str


class ProductItem(BaseModel):
    name: str
    amount: int
    price: int
    total_price: int
    category: str
    type: str


class ScanReceiptResponse(BaseModel):
    products: List[ProductItem]
