
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Generate Car Evaluation Dataset
np.random.seed(11)

n = 1200

buying = np.random.choice(['low', 'med', 'high', 'vhigh'], n)
maint = np.random.choice(['low', 'med', 'high', 'vhigh'], n)
doors = np.random.choice(['2', '3', '4', '5more'], n)
persons = np.random.choice(['2', '4', 'more'], n)
lug_boot = np.random.choice(['small', 'med', 'big'], n)
safety = np.random.choice(['low', 'med', 'high'], n)


# Step 2: Define Category Scores
safety_score = {
    'low': 0,
    'med': 1,
    'high': 2
}

persons_score = {
    '2': 0,
    '4': 1,
    'more': 2
}

buying_score = {
    'low': 3,
    'med': 2,
    'high': 1,
    'vhigh': 0
}


# Step 3: Calculate Total Score
total = (
    np.vectorize(safety_score.get)(safety) * 2
    + np.vectorize(persons_score.get)(persons)
    + np.vectorize(buying_score.get)(buying)
)


# Step 4: Create Car Acceptability Classes
car_class = pd.cut(
    total,
    bins=[-1, 2, 4, 6, 100],
    labels=['unacc', 'acc', 'good', 'vgood']
)


# Step 5: Create DataFrame
df = pd.DataFrame({
    'buying': buying,
    'maint': maint,
    'doors': doors,
    'persons': persons,
    'lug_boot': lug_boot,
    'safety': safety,
    'class': car_class
})

print("Class Distribution:")
print(df['class'].value_counts())


# Step 6: Encode Categorical Features
cat_cols = [
    'buying',
    'maint',
    'doors',
    'persons',
    'lug_boot',
    'safety'
]

encoder = OrdinalEncoder(
    categories=[
        ['low', 'med', 'high', 'vhigh'],
        ['low', 'med', 'high', 'vhigh'],
        ['2', '3', '4', '5more'],
        ['2', '4', 'more'],
        ['small', 'med', 'big'],
        ['low', 'med', 'high']
    ]
)

X = encoder.fit_transform(df[cat_cols])

y = df['class']


# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Step 8: Create Random Forest Model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Step 9: Train the Model
rf.fit(X_train, y_train)


# Step 10: Make Predictions
y_pred = rf.predict(X_test)


# Step 11: Evaluate Model
acc = accuracy_score(y_test, y_pred)

print(f'\nAccuracy: {acc:.4f}')


# Step 12: Display Classification Report
print(
    '\nClassification Report:\n',
    classification_report(y_test, y_pred)
)


# Step 13: Calculate Feature Importance
importances = pd.Series(
    rf.feature_importances_,
    index=cat_cols
).sort_values()


# Step 14: Plot Feature Importance
plt.figure(figsize=(7, 5))

importances.plot(
    kind='barh',
    color='steelblue'
)

plt.title(
    'Feature Importances - Random Forest (Car Evaluation)'
)

plt.xlabel('Importance')

plt.tight_layout()

plt.savefig('rf_car_importance.png', dpi=120)

plt.close()


# Step 15: Project Completion
print("\nRandom Forest Car Evaluation Project Completed!")

print("Generated file: rf_car_importance.png")

