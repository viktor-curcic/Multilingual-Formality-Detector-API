from sklearn.model_selection import train_test_split
import pandas as pd
import os

langs = ['eng', 'fre', 'esp', 'ger']
for lang in langs: 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, f"{lang}/synthetic_data_{lang}.csv"))

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv(os.path.join(current_dir, f"{lang}/train.csv"), index = False)
    test_df.to_csv(os.path.join(current_dir, f"{lang}/test.csv"), index = False)
    val_df.to_csv(os.path.join(current_dir, f"{lang}/val.csv"), index = False)
