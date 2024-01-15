import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('FullThreatsData.csv')

# Assuming you have a list of relevant features
relevant_features = ['OWmean_max_wind', 'OWtotal_precip', 'MeanJ_NE', 'T70p3sum_NE', 'Tempp4avg_NC']  # Replace with actual feature names

# Subset the dataset with relevant features
subset_data = data[relevant_features + ['popgrowth_pva']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(subset_data.drop(['popgrowth_pva'], axis=1), subset_data['popgrowth_pva'], test_size=0.2, random_state=42)

# Create and train the RandomForestRegressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Perform prediction on the test set
predictions = model.predict(X_test)

# Display accuracy metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\nAccuracy Metrics for Random Forest Regressor:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Display feature importance
feature_importance = pd.Series(model.feature_importances_, index=relevant_features).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Display Correlation Matrix for relevant features with larger font size and adjusted layout
correlation_matrix = subset_data.corr()
plt.figure(figsize=(12, 10))
# sns.set(font_scale=1.2)  # Adjust the font size as needed
sns.heatmap(correlation_matrix, annot=True, cmap="mako", fmt=".2f", linewidths=.5, square=True)
plt.title("Correlation Matrix")
plt.show()
