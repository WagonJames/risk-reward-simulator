import pandas as pd
from pathlib import Path

def test_csv_columns():
    csv_path = Path(__file__).resolve().parent.parent / 'testfile.csv'
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    assert list(df.columns) == ['Daily Return', '7-Day MA']

