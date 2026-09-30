import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check


def get_employee_attrition_schema():
    """Defines the strict schema specification for the IBM HR Employee Attrition dataset."""

    return pa.DataFrameSchema({

        # ---------------------------------------------------------
        # Demographic Information
        # ---------------------------------------------------------

        "Age": Column(
            pa.Int,
            Check.ge(18)
        ),

        "BusinessTravel": Column(
            pa.String,
            Check.isin([
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel"
            ])
        ),

        "DailyRate": Column(
            pa.Int,
            Check.ge(0)
        ),

        "Department": Column(
            pa.String,
            Check.isin([
                "Sales",
                "Research & Development",
                "Human Resources"
            ])
        ),

        "DistanceFromHome": Column(
            pa.Int,
            Check.ge(0)
        ),

        "Education": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4, 5])
        ),

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

        "EmployeeCount": Column(
            pa.Int,
            Check.eq(1)
        ),

        "EmployeeNumber": Column(
            pa.Int,
            Check.ge(0),
            nullable=False,
            unique=True
        ),

        "EnvironmentSatisfaction": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4])
        ),

        "Gender": Column(
            pa.String,
            Check.isin([
                "Male",
                "Female"
            ])
        ),

        "JobInvolvement": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4])
        ),

        "JobLevel": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4, 5])
        ),

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

        "JobSatisfaction": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4])
        ),

        "MaritalStatus": Column(
            pa.String,
            Check.isin([
                "Single",
                "Married",
                "Divorced"
            ])
        ),

        "MonthlyIncome": Column(
            pa.Int,
            Check.ge(0)
        ),

        "MonthlyRate": Column(
            pa.Int,
            Check.ge(0)
        ),

        "NumCompaniesWorked": Column(
            pa.Int,
            Check.ge(0)
        ),

        "Over18": Column(
            pa.String,
            Check.eq("Y")
        ),

        "OverTime": Column(
            pa.String,
            Check.isin([
                "Yes",
                "No"
            ])
        ),

        "PercentSalaryHike": Column(
            pa.Int,
            Check.ge(0)
        ),

        "PerformanceRating": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4])
        ),

        "RelationshipSatisfaction": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4])
        ),

        "StandardHours": Column(
            pa.Int,
            Check.eq(80)
        ),

        "StockOptionLevel": Column(
            pa.Int,
            Check.isin([0, 1, 2, 3])
        ),

        "TotalWorkingYears": Column(
            pa.Int,
            Check.ge(0)
        ),

        "TrainingTimesLastYear": Column(
            pa.Int,
            Check.ge(0)
        ),

        "WorkLifeBalance": Column(
            pa.Int,
            Check.isin([1, 2, 3, 4])
        ),

        "YearsAtCompany": Column(
            pa.Int,
            Check.ge(0)
        ),

        "YearsInCurrentRole": Column(
            pa.Int,
            Check.ge(0)
        ),

        "YearsSinceLastPromotion": Column(
            pa.Int,
            Check.ge(0)
        ),

        "YearsWithCurrManager": Column(
            pa.Int,
            Check.ge(0)
        ),

        # ---------------------------------------------------------
        # Target Variable
        # ---------------------------------------------------------

        "Attrition": Column(
            pa.String,
            Check.isin([
                "Yes",
                "No"
            ])
        )

    }, strict=True)


def validate_schema(
    df,
    output_report_name="schema_validation_errors.csv"
):
    """Validates the input DataFrame against the defined schema."""

    print(
        f"[INFO] Validating schema "
        f"(Records: {len(df)})..."
    )

    schema = get_employee_attrition_schema()

    try:

        schema.validate(
            df,
            lazy=True
        )

        print(
            "[SUCCESS] Schema Validation PASSED. "
            "Dataset is clean."
        )

        return True

    except pa.errors.SchemaErrors as err:

        print(
            "[ERROR] Schema Validation FAILED. "
            "Corruptions detected."
        )

        failures = err.failure_cases[
            [
                "schema_context",
                "column",
                "check",
                "failure_case",
                "index"
            ]
        ]

        print(
            failures.to_string()
        )

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        report_path = os.path.join(
            "artifacts",
            output_report_name
        )

        failures.to_csv(
            report_path,
            index=False
        )

        print(
            f"[INFO] Detailed failure report saved to "
            f"'{report_path}'."
        )

        return False


if __name__ == "__main__":

    data_path = (
        "data/raw/"
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    if os.path.exists(data_path):

        raw_df = pd.read_csv(
            data_path
        )

        validate_schema(
            raw_df,
            output_report_name="baseline_validation.csv"
        )

    else:

        print(
            f"[ERROR] Target file not found at: "
            f"{data_path}"
        )