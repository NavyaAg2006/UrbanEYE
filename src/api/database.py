
import pandas as pd
from sqlalchemy import create_engine


# Database connection
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    engine = create_engine(DATABASE_URL)
    return engine


def load_parquet_to_db(parquet_path):
    # Read parquet file
    df = pd.read_parquet(parquet_path)
    print(f"File loaded! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Load into database
    engine = get_engine()
    df.to_sql('crimes', engine, if_exists='replace', index=False)
    print("Data loaded into database successfully!")


if __name__ == "__main__":
    load_parquet_to_db("data/processed/features.parquet")