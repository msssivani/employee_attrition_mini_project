import pandas as pd

from src.validate_data import validate_schema


def simulate_corrupted_data(df):
    """Inject simulated data quality problems into a copy of the dataset."""

    print("[INFO] Injecting simulated faults into Employee Attrition data...")

    corrupted_df = df.copy()

    # 1. Invalid age
    corrupted_df.at[corrupted_df.index[0], "Age"] = -5

    # 2. Invalid department category
    corrupted_df.at[
        corrupted_df.index[1], "Department"
    ] = "Unknown Department"

    # 3. Invalid target value
    corrupted_df.at[
        corrupted_df.index[2], "Attrition"
    ] = "Maybe"

    # 4. Invalid monthly income
    corrupted_df.at[
        corrupted_df.index[3], "MonthlyIncome"
    ] = "Invalid Income"

    # 5. Add an unexpected column
    corrupted_df["unexpected_column"] = 123

    # 6. Remove an expected column
    corrupted_df = corrupted_df.drop(columns=["Gender"])

    return corrupted_df


if __name__ == "__main__":

    # Update this path if your actual CSV filename is different
    data_path = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    try:
        raw_df = pd.read_csv(data_path)

        if len(raw_df) < 4:
            raise ValueError(
                "At least four records are required for corruption simulation."
            )

        corrupted_df = simulate_corrupted_data(raw_df)

        print("\n--- Validating Corrupted Employee Attrition Data ---")

        validate_schema(
            corrupted_df,
            output_report_name="employee_attrition_corruption_errors.csv"
        )

    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")

    except (KeyError, ValueError) as err:
        print(f"[ERROR] Could not simulate corruption: {err}")