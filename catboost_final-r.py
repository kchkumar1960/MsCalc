# Training with CatBoostRegressor
# Uses all data at once
# No GridSearchCV but hyperparameters obtained from GridSearchCV run are used
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor


# A pause function
def pause():
    print('Press <ENTER> key to start...')
    input()


pause()

# Load data
file_path = "CombinedData_filtered_mf.xlsx"
df = pd.read_excel(file_path)

# Shuffle rows
filtered_df = df.sample(frac=1, random_state=101)

# Features and target
X = filtered_df.drop(columns=
                     ['Ms(K)', 'O',
                      'cbya',
                      'ar_wsd', 'ar_rms', 'arFe_rms',
                      'en_rms', 'en_wsd', 'enFe_rms',
                      'ebya_bar', 'vec_bar',
                      'Sconf', 'Sconf_fcc',
                      'cte_bar'
                      ])
y = filtered_df['Ms(K)']
print("Features:", X.columns)

# Apply scaling once
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "x_scaler.pkl")
print("Scaler saved as x_scaler.pkl")

# Hyperparameters are from 5-fold CV run 
best_params = {'bagging_temperature': 0, 'depth': 4,
               'grow_policy': 'SymmetricTree', 'iterations': 2200,
               'l2_leaf_reg': 0, 'learning_rate': 0.024, 'random_strength': 1}


# globals().update(best_params)  # Converts dictionary to variables


# Train final model on full dataset with best params
final_model = CatBoostRegressor(
    **best_params,
    random_state=42,
    verbose=0
)
final_model.fit(X_scaled, y)

# Save final model
final_model.save_model("catboost_final-r.cbm")
print("Final model saved as catboost_final-r.cbm")

# Predictions on training data (since no holdout set)
y_pred = final_model.predict(X_scaled)

# Metrics
rmse = sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("\n=== Final Model Metrics on Full Data ===")
print(f"RMSE: {rmse:.3f}")
print(f"R2: {r2:.3f}")
print(f"MAE: {mae:.3f}")

# Parity plot
plt.figure(figsize=(7, 6))
plt.scatter(y, y_pred, alpha=0.6, color='blue', label='Data points')

# Diagonal y = x line
min_val = min(min(y), min(y_pred))
max_val = max(max(y), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')

plt.xlabel("Actual values (in K)")
plt.ylabel("Predicted values (in K)")
plt.title("Predicted vs Actual Values (Final Model)")

# Add metrics text box
box_text = f"RMSE: {rmse:.2f}\nR²: {r2:.2f}\nMAE: {mae:.2f}"
plt.text(0.05, 0.95, box_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

plt.legend()
plt.tight_layout()
plt.show()
