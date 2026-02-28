import pandas as pd
import os

csv_path = r'D:\PROJECTS\Rain Predict\batch_template.csv'
train_data_dir = r'D:\PROJECTS\Rain Predict\Train Data'

df = pd.read_csv(csv_path)
expected_files = set(df['output_name'].dropna().astype(str).tolist())

actual_files = set()
for f in os.listdir(train_data_dir):
    if f.endswith('.csv'):
        actual_files.add(f[:-4])

missing_files = expected_files - actual_files
extra_files = actual_files - expected_files

print(f"Expected: {len(expected_files)}")
print(f"Actual:   {len(actual_files)}")
print(f"\nMissing files ({len(missing_files)}):")
for f in sorted(missing_files):
    print(f"  - {f}")

print(f"\nExtra files ({len(extra_files)}):")
for f in sorted(extra_files):
    print(f"  - {f}")
