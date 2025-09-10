import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("Advertising.csv")
if "" in df.columns:
    df = df.drop(columns=[""])

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0,0])
axes[0,0].set_title("Correlation Heatmap of Features")

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

axes[0,1].scatter(y_test, y_pred, alpha=0.7, color="blue")
axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
axes[0,1].set_title("Actual vs Predicted Sales")
axes[0,1].set_xlabel("Actual Sales")
axes[0,1].set_ylabel("Predicted Sales")

coefficients = (
    pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    .sort_values(by="Coefficient", ascending=False)
)
sns.barplot(data=coefficients, x="Coefficient", y="Feature", palette="viridis", ax=axes[1,0])
axes[1,0].set_title("Feature Importance in Sales Prediction")

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1,1], color="purple")
axes[1,1].axhline(0, color="red", linestyle="--")
axes[1,1].set_title("Residual Plot")
axes[1,1].set_xlabel("Predicted Sales")
axes[1,1].set_ylabel("Residuals")

plt.tight_layout()
plt.show()
