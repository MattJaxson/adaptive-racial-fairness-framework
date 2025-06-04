import pandas as pd

def load_data(source, file_type='csv', table_name=None, db_connection=None):
    """
    Load data from different sources (CSV, SQL).
    
    Args:
        source (str): Path to the file or SQL query string.
        file_type (str): 'csv' or 'sql'.
        table_name (str): If loading from a database, provide table name.
        db_connection: SQLAlchemy connection engine or similar.
    
    Returns:
        pd.DataFrame: Loaded and preprocessed data.
    """
    if file_type == 'csv':
        try:
            df = pd.read_csv(source)
            print(f"Loaded {len(df)} rows from {source}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    elif file_type == 'sql':
        if db_connection is None or table_name is None:
            raise ValueError("Provide db_connection and table_name for SQL loading.")
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, db_connection)
            print(f"Loaded {len(df)} rows from table {table_name}")
        except Exception as e:
            print(f"Error loading SQL data: {e}")
            return None
    else:
        raise ValueError("file_type must be 'csv' or 'sql'.")

    # Basic data cleaning
    df = df.dropna(subset=['race', 'hired'])
    print("Dropped rows with missing race or hired values.")

    # Fill missing feature values with median (for numerical columns)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median {median_val}.")
    
    return df
