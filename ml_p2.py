
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Generate Synthetic Uber Ride Data
np.random.seed(42)

n = 1000

pickup_lat = np.random.uniform(40.70, 40.80, n)
pickup_lon = np.random.uniform(-74.02, -73.93, n)

drop_lat = np.random.uniform(40.70, 40.80, n)
drop_lon = np.random.uniform(-74.02, -73.93, n)

passenger_count = np.random.randint(1, 5, n)


# Step 2: Calculate Distance Using Haversine Formula
def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2)**2
        + np.cos(phi1) * np.cos(phi2)
        * np.sin(dlambda / 2)**2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


distance_km = haversine(
    pickup_lat,
    pickup_lon,
    drop_lat,
    drop_lon
)


# Step 3: Generate Fare Amount
fare = (
    2.5
    + 1.8 * distance_km
    + 0.3 * passenger_count
    + np.random.normal(0, 1.5, n)
)


# Step 4: Create DataFrame
df = pd.DataFrame({
    'distance_km': distance_km,
    'passenger_count': passenger_count,
    'fare_amount': fare
})


# Step 5: Introduce Outliers
outlier_idx = np.random.choice(
    df.index,
    size=15,
    replace=False
)

df.loc[outlier_idx, 'fare_amount'] = (
    df.loc[outlier_idx, 'fare_amount']
    * np.random.uniform(6, 10, 15)
)

print('Rows before outlier removal:', len(df))


# Step 6: Remove Outliers Using IQR
Q1, Q3 = df['fare_amount'].quantile([0.25, 0.75])

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[
    (df['fare_amount'] >= lower)
    & (df['fare_amount'] <= upper)
].copy()

print('Rows after outlier removal:', len(df_clean))


# Step 7: Check Correlation
corr = df_clean.corr()

print('\nCorrelation Matrix:\n', corr)


# Step 8: Plot Correlation Heatmap
plt.figure(figsize=(5, 4))

plt.imshow(
    corr,
    cmap='coolwarm',
    vmin=-1,
    vmax=1
)

plt.xticks(
    range(len(corr)),
    corr.columns,
    rotation=45
)

plt.yticks(
    range(len(corr)),
    corr.columns
)

plt.colorbar()

plt.title('Correlation Heatmap')

plt.tight_layout()

plt.savefig('uber_correlation.png', dpi=120)

plt.close()


# Step 9: Split Data into Training and Testing
X = df_clean[['distance_km', 'passenger_count']]

y = df_clean['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 10: Standardize Features
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)

X_test_s = scaler.transform(X_test)


# Step 11: Create Regression Models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01)
}

results = []


# Step 12: Train and Evaluate Models
for name, model in models.items():

    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)

    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    r2 = r2_score(y_test, preds)

    results.append({
        'Model': name,
        'RMSE': rmse,
        'R2': r2
    })

    print(
        f'{name}: RMSE={rmse:.3f}, R2={r2:.4f}'
    )


# Step 13: Display Model Comparison
results_df = pd.DataFrame(results)

print('\nModel Comparison:\n', results_df)


# Step 14: Project Completion
print('\nUber Fare Prediction Project Completed!')

print('Generated file: uber_correlation.png')
