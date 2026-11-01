
import os
import sys
import pandas as pd
import yaml
import pandera.pandas as pa
from pandera import Column, Check


def get_employee_attrition_schema():

    schema = pa.DataFrameSchema(
        {
            "Age": Column(int, Check.ge(18)),
            "BusinessTravel": Column(
                str,
                Check.isin([
                    "Travel_Rarely",
                    "Travel_Frequently",
                    "Non-Travel"
                ])
            ),
            "DailyRate": Column(int, Check.ge(0)),
            "Department": Column(
                str,
                Check.isin([
                    "Sales",
                    "Research & Development",
                    "Human Resources"
                ])
            ),
            "DistanceFromHome": Column(int, Check.ge(0)),
            "Education": Column(int, Check.isin([1, 2, 3, 4, 5])),
            "EducationField": Column(
                str,
                Check.isin([
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                    "Other"
                ])
            ),
            "EmployeeCount": Column(int, Check.eq(1)),
            "EmployeeNumber": Column(
                int,
                Check.ge(0),
                unique=True
            ),
            "EnvironmentSatisfaction": Column(
                int, Check.isin([1, 2, 3, 4])
            ),
            "Gender": Column(str, Check.isin(["Male", "Female"])),
            "HourlyRate": Column(int, Check.ge(0)),
            "JobInvolvement": Column(int, Check.isin([1, 2, 3, 4])),
            "JobLevel": Column(int, Check.isin([1, 2, 3, 4, 5])),
            "JobRole": Column(
                str,
                Check.isin([
                    "Healthcare Representative",
                    "Human Resources",
                    "Laboratory Technician",
                    "Manager",
                    "Manufacturing Director",
                    "Research Director",
                    "Research Scientist",
                    "Sales Executive",
                    "Sales Representative"
                ])
            ),
            "JobSatisfaction": Column(
                int, Check.isin([1, 2, 3, 4])
            ),
            "MaritalStatus": Column(
                str,
                Check.isin(["Single", "Married", "Divorced"])
            ),
            "MonthlyIncome": Column(int, Check.ge(0)),
            "MonthlyRate": Column(int, Check.ge(0)),
            "NumCompaniesWorked": Column(int, Check.ge(0)),
            "Over18": Column(str, Check.eq("Y")),
            "OverTime": Column(str, Check.isin(["Yes", "No"])),
            "PercentSalaryHike": Column(int, Check.ge(0)),
            "PerformanceRating": Column(
                int, Check.isin([1, 2, 3, 4])
            ),
            "RelationshipSatisfaction": Column(
                int, Check.isin([1, 2, 3, 4])
            ),
            "StandardHours": Column(int, Check.eq(80)),
            "StockOptionLevel": Column(
                int, Check.isin([0, 1, 2, 3])
            ),
            "TotalWorkingYears": Column(int, Check.ge(0)),
            "TrainingTimesLastYear": Column(int, Check.ge(0)),
            "WorkLifeBalance": Column(
                int, Check.isin([1, 2, 3, 4])
            ),
            "YearsAtCompany": Column(int, Check.ge(0)),
            "YearsInCurrentRole": Column(int, Check.ge(0)),
            "YearsSinceLastPromotion": Column(int, Check.ge(0)),
            "YearsWithCurrManager": Column(int, Check.ge(0)),
            "Attrition": Column(str, Check.isin(["Yes", "No"]))
        },
        strict=True
    )

    return schema


def validate_schema(df, output_report_name="baseline_validation.csv"):

    print("[INFO] Starting Employee Attrition Schema Validation...")

    schema = get_employee_attrition_schema()

    try:
        schema.validate(df, lazy=True)

        print("[SUCCESS] Schema validation passed!")
        return True

    except pa.errors.SchemaErrors as e:

        print("[ERROR] Schema validation failed!")

        os.makedirs("artifacts", exist_ok=True)

        failure_cases = e.failure_cases

        report_path = os.path.join(
            "artifacts",
            output_report_name
        )

        failure_cases.to_csv(report_path, index=False)

        print(f"[INFO] Failure report saved to: {report_path}")
        print(failure_cases)

        return False


if __name__ == "__main__":

    print("[INFO] Loading configuration...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]

    print(f"[INFO] Dataset path: {data_path}")

    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)

    passed = validate_schema(
        raw_df,
        output_report_name="baseline_validation.csv"
    )

    sys.exit(0 if passed else 1)
