import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Create Synthetic Dataset
np.random.seed(42)
n_rows = 1000
data = {
    'Hour': np.random.randint(0, 24, n_rows),
    'Day_of_Week': np.random.randint(0, 7, n_rows),
    'Traffic_Density': np.random.uniform(0.1, 1.0, n_rows),
    'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Snowy'], n_rows),
    'Road_Surface': np.random.choice(['Dry', 'Wet', 'Icy'], n_rows),
    'Speed_Limit': np.random.choice([30, 50, 70, 90, 110], n_rows)
}

# Target: Accident Severity (0: Safe, 1: Minor, 2: Major)
# Logic: High density + bad weather + high speed = high risk
risk_score = (data['Traffic_Density'] * 5 + 
              (data['Weather'] == 'Rainy').astype(int) * 2 +
              (data['Weather'] == 'Snowy').astype(int) * 4 +
              (data['Speed_Limit'] / 30))

data['Accident_Severity'] = pd.cut(risk_score, bins=[-np.inf, 4, 8, np.inf], labels=[0, 1, 2]).astype(int)
df = pd.DataFrame(data)

# 2. Preprocessing
le_weather = LabelEncoder()
le_surface = LabelEncoder()
df['Weather'] = le_weather.fit_transform(df['Weather'])
df['Road_Surface'] = le_surface.fit_transform(df['Road_Surface'])

# 3. Train Model
X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save everything
joblib.dump(model, 'traffic_model.pkl')
joblib.dump(le_weather, 'le_weather.pkl')
joblib.dump(le_surface, 'le_surface.pkl')
print("✅ Model Trained & Saved as traffic_model.pkl!")