
import pandas as pd
from sqlalchemy import create_engine


# Database connection
DATABASE_URL = "postgresql://postgres:shri2006@localhost:5432/urbaneye"


def get_engine():
    engine = create_engine(DATABASE_URL)
    return engine


def load_csv_to_db(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"CSV loaded! Rows: {len(df)}, Columns: {list(df.columns)}")

    # Load into database
    engine = get_engine()
    df.to_sql('crimes', engine, if_exists='append', index=False)
    print("Data loaded into database successfully!")


if __name__ == "__main__":
    load_csv_to_db("data/raw/sample_crimes.csv")