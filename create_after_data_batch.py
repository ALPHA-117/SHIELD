import pandas as pd
from datetime import timedelta

# Read the existing template
df = pd.read_csv('batch_template.csv')

# Create new dataframe for the next 15 days
new_df = df.copy()

# Convert dates to datetime objects for calculation
new_df['start_date'] = pd.to_datetime(df['end_date'])
new_df['end_date'] = new_df['start_date'] + timedelta(days=15)

# Format dates back to string
new_df['start_date'] = new_df['start_date'].dt.strftime('%Y-%m-%d')
new_df['end_date'] = new_df['end_date'].dt.strftime('%Y-%m-%d')

# Append _after_data to the output name
new_df['output_name'] = df['output_name'] + '_after_data'

# Save to new CSV
new_df.to_csv('batch_after_data.csv', index=False)
print("Created batch_after_data.csv with 15 days of subsequent dates.")
