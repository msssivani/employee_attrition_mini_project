import os
import pandas as pd
import pandera.pandas as pa

from pandera import Check, Column


def get_employee_attrition_schema():

    schema = pa.DataFrameSchema({

        # Employee information
        "Age": Column(pa.Int, Check.in_range(18, 100)),
        "Attrition": Column(pa.String, Check.isin(["Yes", "No"])),
        "BusinessTravel": Column(
            pa.String,
            Check.isin([
                "Non-Travel",
                "Travel_Frequently",
                "Travel_Rarely"
            ])
        ),
        "DailyRate": Column(pa.Int, Check.ge(0)),
        "Department": Column(
            pa.String,
            Check.isin([
                "Sales",
                "Research & Development",
                "Human Resources"
            ])
        ),
        "DistanceFromHome": Column(pa.Int, Check.ge(0)),
        "Education": Column(pa.Int, Check.in_range(1, 5)),
        "EducationField": Column(
            pa.String,
            Check.isin([
                "Life Sciences",
                "Medical",
                "Marketing",
                "Technical Degree",
                "Human Resources",
                "Other"
            ])
        ),

        # Columns that are constant in the original dataset
        "EmployeeCount": Column(pa.Int, Check.eq(1)),
        "EmployeeNumber": Column(pa.Int, Check.ge(0)),

        # Satisfaction and job information
        "EnvironmentSatisfaction": Column(pa.Int, Check.in_range(1, 4)),
        "Gender": Column(pa.String, Check.isin(["Male", "Female"])),
        "HourlyRate": Column(pa.Int, Check.in_range(1, 100)),
        "JobInvolvement": Column(pa.Int, Check.in_range(1, 4)),
        "JobLevel": Column(pa.Int, Check.ge(1)),
        "JobRole": Column(
            pa.String,
            Check.isin([
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ])
        ),
        "JobSatisfaction": Column(pa.Int, Check.in_range(1, 4)),
        "MaritalStatus": Column(
            pa.String,
            Check.isin(["Single", "Married", "Divorced"])
        ),
        "MonthlyIncome": Column(pa.Int, Check.ge(0)),
        "MonthlyRate": Column(pa.Int, Check.ge(0)),
        "NumCompaniesWorked": Column(pa.Int, Check.ge(0)),

        # Additional employee attributes
        "Over18": Column(pa.String, Check.eq("Y")),
        "OverTime": Column(pa.String, Check.isin(["Yes", "No"])),
        "PercentSalaryHike": Column(pa.Int, Check.ge(0)),
        "PerformanceRating": Column(pa.Int, Check.in_range(1, 4)),
        "RelationshipSatisfaction": Column(pa.Int, Check.in_range(1, 4)),
        "StandardHours": Column(pa.Int, Check.eq(80)),
        "StockOptionLevel": Column(pa.Int, Check.in_range(0, 3)),
        "TotalWorkingYears": Column(pa.Int, Check.ge(0)),
        "TrainingTimesLastYear": Column(pa.Int, Check.ge(0)),
        "WorkLifeBalance": Column(pa.Int, Check.in_range(1, 4)),
        "YearsAtCompany": Column(pa.Int, Check.ge(0)),
        "YearsInCurrentRole": Column(pa.Int, Check.ge(0)),
        "YearsSinceLastPromotion": Column(pa.Int, Check.ge(0)),
        "YearsWithCurrManager": Column(pa.Int, Check.ge(0))

    }, strict=True)

    return schema


def validate_schema(
    df,
    output_report_name="schema_validation_errors.csv"
):

    print(f"[INFO] Validating schema (Records: {len(df)})...")

    schema = get_employee_attrition_schema()

    try:
        schema.validate(df, lazy=True)

        print("[SUCCESS] Employee Attrition schema validation PASSED.")
        return True

    except pa.errors.SchemaErrors as err:

        print("[ERROR] Schema Validation FAILED.")

        failures = err.failure_cases[
            [
                "schema_context",
                "column",
                "check",
                "failure_case",
                "index"
            ]
        ]

        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)

        report_path = os.path.join(
            "artifacts",
            output_report_name
        )

        failures.to_csv(report_path, index=False)

        print(f"[INFO] Failure report saved to '{report_path}'.")

        return False