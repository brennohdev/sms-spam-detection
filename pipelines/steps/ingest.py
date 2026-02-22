import pandas as pd
import os

class DataIngestor:
    def execute(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}")
        
        df = pd.read_csv(path, encoding='latin-1')
        # Standardize our columns
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
        return df