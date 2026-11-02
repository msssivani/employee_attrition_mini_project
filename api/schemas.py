
from typing import Literal

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):

    # Numerical features
    EmployeeNumber: int = Field(ge=0)
    Age: int = Field(ge=18)
    DailyRate: float = Field(ge=0)
    DistanceFromHome: int = Field(ge=0)
    HourlyRate: int = Field(ge=0, le=100)
    MonthlyIncome: float = Field(ge=0)
    MonthlyRate: float = Field(ge=0)
    NumCompaniesWorked: int = Field(ge=0)
    PercentSalaryHike: int = Field(ge=0)
    TotalWorkingYears: int = Field(ge=0)
    TrainingTimesLastYear: int = Field(ge=0)
    YearsAtCompany: int = Field(ge=0)
    YearsInCurrentRole: int = Field(ge=0)
    YearsSinceLastPromotion: int = Field(ge=0)
    YearsWithCurrManager: int = Field(ge=0)

    # Ordinal features
    Education: Literal[1, 2, 3, 4, 5]

    EnvironmentSatisfaction: Literal[1, 2, 3, 4]

    JobInvolvement: Literal[1, 2, 3, 4]

    JobLevel: Literal[1, 2, 3, 4, 5]

    JobSatisfaction: Literal[1, 2, 3, 4]

    RelationshipSatisfaction: Literal[1, 2, 3, 4]

    StockOptionLevel: Literal[0, 1, 2, 3]

    WorkLifeBalance: Literal[1, 2, 3, 4]

    PerformanceRating: Literal[3, 4]

    # Categorical features
    BusinessTravel: Literal[
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]

    Department: Literal[
        "Sales",
        "Research & Development",
        "Human Resources"
    ]

    EducationField: Literal[
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Other",
        "Human Resources"
    ]

    Gender: Literal["Male", "Female"]

    JobRole: Literal[
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]

    MaritalStatus: Literal[
        "Single",
        "Married",
        "Divorced"
    ]

    OverTime: Literal["Yes", "No"]


class PredictionResponse(BaseModel):

    prediction: Literal[0, 1]

    attrition_probability: float = Field(ge=0, le=1)


class ErrorDetail(BaseModel):

    field: str
    message: str


class ErrorResponse(BaseModel):

    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):

    employees: list[EmployeeInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):

    count: int
    predictions: list[PredictionResponse]
