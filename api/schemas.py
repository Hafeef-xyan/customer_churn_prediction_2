from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    
    AccountWeeks: float = Field(ge=0)
    ContractRenewal: int = Field(ge=0, le=1)
    DataPlan: int = Field(ge=0, le=1)
    DataUsage: float = Field(ge=0)
    CustServCalls: int = Field(ge=0)
    DayMins: float = Field(ge=0)
    DayCalls: int = Field(ge=0)
    MonthlyCharge: float = Field(ge=0)
    OverageFee: float = Field(ge=0)
