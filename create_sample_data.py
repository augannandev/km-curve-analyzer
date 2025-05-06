import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create sample survival data for Treatment A
time_months = np.arange(0, 60, 1)  # 0 to 60 months
survival_prob = 100 * np.exp(-0.02 * time_months)  # Exponential decay starting at 100%

# Create dataframe
df = pd.DataFrame({
    'time_months': time_months,
    'curve_name': 'Treatment A',
    'survival_prob': survival_prob
})

# Save to Excel
df.to_excel(os.path.join(data_dir, 'treatment_a.xlsx'), index=False)

# Create sample data for Control Group
time_months = np.arange(0, 60, 1)  # 0 to 60 months
survival_prob = 100 * np.exp(-0.04 * time_months)  # Faster decay for control group

# Create dataframe
df = pd.DataFrame({
    'time_months': time_months,
    'curve_name': 'Control',
    'survival_prob': survival_prob
})

# Save to Excel
df.to_excel(os.path.join(data_dir, 'control.xlsx'), index=False)

print("Sample Excel files created successfully!")
