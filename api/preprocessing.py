
import pandas as pd


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw employee data before attrition prediction.
    """

    df = df.copy()
    employee_numbers = (
        df["EmployeeNumber"].copy()
        if "EmployeeNumber" in df.columns
        else None
    )
    # Remove EmployeeNumber because it is an identifier,
    # not a feature used by the model.
    if "EmployeeNumber" in df.columns:
        df = df.drop(columns=["EmployeeNumber"])

    # Remove target column if someone accidentally sends it.
    if "Attrition" in df.columns:
        df = df.drop(columns=["Attrition"])

    # Remove columns that were excluded during model training.
    columns_to_drop = [
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]

    existing_columns = [
        col for col in columns_to_drop if col in df.columns
    ]

    if existing_columns:
        df = df.drop(columns=existing_columns)

    return df,employee_numbers
