from typing import Optional
from pydantic import BaseModel, Field

class ExtractionResult(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    date: Optional[str] = Field(None, description="Invoice date")
    total_amount: Optional[str] = Field(None, description="Total amount on the invoice")
    vendor_name: Optional[str] = Field(None, description="Name of the vendor")
    customer_name: Optional[str] = Field(None, description="Name of the customer") 