import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / 'data' / 'customer_churn_data.csv'

OUTPUT_DIR = BASE_DIR / 'data' / 'processed_data'

initial_path = OUTPUT_DIR / "initial_train.csv"
retrain_path = OUTPUT_DIR / "retraining_data.csv"

def split_data():

    if initial_path.exists() and retrain_path.exists():
        print("Datasets are already split. Skipping data splitting.")
        return
    
    df = pd.read_csv(DATA_PATH)

    # check dataset contains enough rows
    if len(df) < 2000:
        raise ValueError("Dataset must contain 2000 above rows... check dataset..")

    # first 2000 data for first training
    initial_data = df.iloc[:2000].copy()

    # remaining data for retraining
    retrain_data = df.iloc[2000:].copy()

    # save datasets
    initial_data.to_csv(
        initial_path,
        index=False
    )
    retrain_data.to_csv(
        retrain_path,
        index=False
    )

    print("\n Data splitting completed..")
    print(f"initial training data contains {len(initial_data)} rows.")
    print(f"retrain data contains {len(retrain_data)} rows.")

if __name__ == "__main__":
    split_data()