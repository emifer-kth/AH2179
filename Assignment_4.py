import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


SEED = 42


def reset_seed():
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


reset_seed()


url = "https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_7_Neural_networks/Exercise7BikeSharing.csv"

df = pd.read_csv(url)

print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset shape:")
print(df.shape)
print("\nDataset information:")
df.info()


target = "cnt"

features = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "weathersit",
    "hr",
    "weekday",
    "workingday",
    "holiday",
    "season",
    "yr"
]

X = df[features].copy()
y = df[target].astype(float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)


linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)

y_pred_linear = linear_model.predict(X_test_scaled)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("LINEAR REGRESSION RESULTS")
print(f"Mean Absolute Error: {mae_linear:.4f}")
print(f"Mean Squared Error: {mse_linear:.4f}")
print(f"Root Mean Squared Error: {rmse_linear:.4f}")
print(f"R-squared: {r2_linear:.4f}")


reset_seed()

baseline_model = Sequential()
baseline_model.add(Input(shape=(len(features),)))
baseline_model.add(Dense(32, activation="relu"))
baseline_model.add(Dense(64, activation="relu"))
baseline_model.add(Dense(1))
baseline_model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

print("BASELINE NEURAL NETWORK")

baseline_model.summary()

hist_baseline = baseline_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=200,
    batch_size=32,
    verbose=1)

sns.set()

training_mae_baseline = hist_baseline.history["mae"]
validation_mae_baseline = hist_baseline.history["val_mae"]

epochs_baseline = range(1, len(training_mae_baseline) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_baseline, training_mae_baseline, "-", label="Training MAE")
plt.plot(epochs_baseline, validation_mae_baseline, ":", label="Validation MAE")
plt.title("Baseline NN - Training and Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


y_pred_baseline = baseline_model.predict(X_test_scaled, verbose=0).reshape(-1)

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print("BASELINE NN RESULTS ")
print("Architecture: 32 -> 64")
print("Dropout: 0.0")
print("Learning rate: 0.001")
print(f"Mean Absolute Error: {mae_baseline:.4f}")
print(f"Mean Squared Error: {mse_baseline:.4f}")
print(f"Root Mean Squared Error: {rmse_baseline:.4f}")
print(f"R-squared: {r2_baseline:.4f}")


def build_model(hidden_layers, dropout_rate=0.0):

    reset_seed()

    model = Sequential()

    model.add(Input(shape=(len(features),)))

    for neurons in hidden_layers:
        model.add(Dense(neurons, activation="relu"))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mae", metrics=["mae"])

    return model


network_sizes = [
    (16, 32),
    (32, 64),
    (64, 64),
    (64, 128),
    (128, 64)
]

size_results = []

best_architecture = None
best_architecture_mae = float("inf")

print("NETWORK SIZE TUNING")


for hidden_layers in network_sizes:

    print(f"\nTesting architecture: {hidden_layers}")

    model = build_model(hidden_layers, dropout_rate=0.0)

    early_stop = EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True, mode="min")

    reduce_lr = ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        mode="min",
        verbose=0
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    best_val_mae = min(history.history["val_mae"])

    size_results.append({
        "Architecture": str(hidden_layers),
        "Dropout": 0.0,
        "Validation MAE": best_val_mae
    })

    print(f"Best validation MAE: {best_val_mae:.4f}")

    if best_val_mae < best_architecture_mae:
        best_architecture_mae = best_val_mae
        best_architecture = hidden_layers


size_results_df = pd.DataFrame(size_results)
size_results_df = size_results_df.sort_values("Validation MAE")

print(" NETWORK SIZE RESULTS")
print(size_results_df.to_string(index=False))
print("\nBest architecture:", best_architecture)
print(f"Best validation MAE: {best_architecture_mae:.4f}")


dropout_rates = [0.0,0.1,0.2,0.3]
dropout_results = []
best_dropout = None
best_dropout_mae = float("inf")

print("DROPOUT TUNING")


for dropout_rate in dropout_rates:

    print(f"\nTesting dropout: {dropout_rate}")

    model = build_model(best_architecture, dropout_rate=dropout_rate)

    early_stop = EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True, mode="min")

    reduce_lr = ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        mode="min",
        verbose=0
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=200,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    best_val_mae = min(history.history["val_mae"])

    dropout_results.append({
        "Architecture": str(best_architecture),
        "Dropout": dropout_rate,
        "Validation MAE": best_val_mae
    })

    print(f"Best validation MAE: {best_val_mae:.4f}")

    if best_val_mae < best_dropout_mae:
        best_dropout_mae = best_val_mae
        best_dropout = dropout_rate


dropout_results_df = pd.DataFrame(dropout_results)
dropout_results_df = dropout_results_df.sort_values("Validation MAE")

print("DROPOUT RESULTS")
print(dropout_results_df.to_string(index=False))
print("\nBest architecture:", best_architecture)
print("Best dropout:", best_dropout)
print(f"Best validation MAE: {best_dropout_mae:.4f}")
print("Learning rate: 0.001")


final_model = build_model(best_architecture, dropout_rate=best_dropout)

print("FINAL NEURAL NETWORK")
print("Architecture:", best_architecture)
print("Dropout:", best_dropout)
print("Learning rate: 0.001")

final_model.summary()


early_stop = EarlyStopping(
    monitor="val_mae",
    patience=5,
    restore_best_weights=True,
    mode="min")

reduce_lr = ReduceLROnPlateau(
    monitor="val_mae",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    mode="min",
    verbose=1)

filepath = "weights.best.keras"

checkpoint = ModelCheckpoint(
    filepath,
    monitor="val_mae",
    verbose=1,
    save_best_only=True,
    mode="min")


hist_final = final_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1)

training_mae_final = hist_final.history["mae"]
validation_mae_final = hist_final.history["val_mae"]

epochs_final = range(1, len(training_mae_final) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_final, training_mae_final, "-", label="Training MAE")
plt.plot(epochs_final, validation_mae_final, ":", label="Validation MAE")
plt.title("Best NN - Training and Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

best_model = load_model(filepath)
y_pred_best = best_model.predict(X_test_scaled, verbose=0).reshape(-1)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print("BEST NEURAL NETWORK RESULTS")
print("Architecture:", best_architecture)
print("Dropout:", best_dropout)
print("Learning rate: 0.001")
print(f"Validation MAE: {best_dropout_mae:.4f}")
print(f"Mean Absolute Error: {mae_best:.4f}")
print(f"Mean Squared Error: {mse_best:.4f}")
print(f"Root Mean Squared Error: {rmse_best:.4f}")
print(f"R-squared: {r2_best:.4f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)

minimum = min(y_test.min(), y_pred_best.min())
maximum = max(y_test.max(), y_pred_best.max())

plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=2)
plt.xlabel("Actual Bike Demand")
plt.ylabel("Predicted Bike Demand")
plt.title("Actual vs Predicted Bike-Sharing Demand")
plt.tight_layout()
plt.show()


results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "MAE": mae_linear,
        "MSE": mse_linear,
        "RMSE": rmse_linear,
        "R2": r2_linear
    },
    {
        "Model": "Baseline Neural Network",
        "MAE": mae_baseline,
        "MSE": mse_baseline,
        "RMSE": rmse_baseline,
        "R2": r2_baseline
    },
    {
        "Model": "Best Tuned Neural Network",
        "MAE": mae_best,
        "MSE": mse_best,
        "RMSE": rmse_best,
        "R2": r2_best
    }
])

print("FINAL MODEL COMPARISON")
print(results.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["RMSE"])
plt.ylabel("RMSE")
plt.title("RMSE Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(size_results_df["Architecture"], size_results_df["Validation MAE"])
plt.xlabel("Network Architecture")
plt.ylabel("Validation MAE")
plt.title("Network Size Tuning")
plt.tight_layout()
plt.show()

dropout_plot = dropout_results_df.sort_values("Dropout")

plt.figure(figsize=(8, 5))
plt.plot(dropout_plot["Dropout"], dropout_plot["Validation MAE"], marker="o")
plt.xlabel("Dropout Rate")
plt.ylabel("Validation MAE")
plt.title("Dropout Tuning")
plt.tight_layout()
plt.show()