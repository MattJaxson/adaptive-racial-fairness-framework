from racial_bias_score import calculate_racial_bias_score
import pandas as pd

# Example data
data = {
    'race': ['Black', 'White', 'Latinx', 'Black', 'White', 'Latinx'],
    'outcome': [0, 1, 0, 0, 1, 1]  # 1 = approved, 0 = denied
}

df = pd.DataFrame(data)
results = calculate_racial_bias_score(df)
print(results)
