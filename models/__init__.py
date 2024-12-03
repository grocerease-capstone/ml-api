from typing import List
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str


class ProductItemDetail(BaseModel):
    type: str
    category_index: int
    category_probability: float


class ProductItem(BaseModel):
    name: str
    amount: int
    price: int
    total_price: int
    detail: ProductItemDetail


class ScanReceiptResponse(BaseModel):
    products: List[ProductItem]
