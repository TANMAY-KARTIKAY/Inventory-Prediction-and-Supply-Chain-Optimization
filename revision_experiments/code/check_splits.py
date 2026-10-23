import pandas as pd

for name in ['train_split.csv', 'validation_split.csv', 'test_split.csv']:
    path = f'Temporal Split(6)/{name}'
    df = pd.read_csv(path, nrows=3)
    print(f"=== {name} ===")
    print("Columns:", df.columns.tolist()[:10])
    df_date = pd.read_csv(path, usecols=['date'])
    print(f"Date range: {df_date['date'].min()} to {df_date['date'].max()} ({len(df_date)} rows)")
