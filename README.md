# Codes

## PSI Code Usage
```py
import psi
import pandas as pd
df = pd.read_csv('some_data.csv')
psi_metrics = psi.PSI('date', 'category', 'count')
psi_metrics.fit(df)
psi_metrics.transform(df)
```
