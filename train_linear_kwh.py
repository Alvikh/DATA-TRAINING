import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Load Data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath, parse_dates=['measured_at'])
        print(f"\n📁 Data dari {filepath} berhasil dimuat.")
        print(data.head(3))

        required_columns = ['measured_at', 'voltage', 'current', 'power', 'energy',
                            'frequency', 'power_factor', 'temperature', 'humidity']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {missing_cols}")
        return data
    except Exception as e:
        print(f"❌ Gagal memuat data: {e}")
        exit()

# Feature Engineering
def prepare_data(data, target='power', log_transform=True, remove_outliers=True, poly=True):
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' tidak ditemukan.")

    data = data.copy()
    data = data[data[target] > 1]  # Buang noise power kecil

    # Fitur waktu
    data['hour'] = data['measured_at'].dt.hour
    data['day_of_week'] = data['measured_at'].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)

    # Fitur interaksi
    data['volt_curr'] = data['voltage'] * data['current']
    data['curr_squared'] = data['current'] ** 2
    data['temp_humid'] = data['temperature'] * data['humidity']

    if remove_outliers:
        z = np.abs((data[target] - data[target].mean()) / data[target].std())
        data = data[z < 3]

    y_raw = data[target]
    y = np.log1p(y_raw) if log_transform else y_raw

    base_features = ['voltage', 'current', 'energy', 'frequency', 'power_factor',
                     'temperature', 'humidity', 'volt_curr', 'curr_squared', 'temp_humid']
    time_features = ['sin_hour', 'cos_hour', 'is_weekend']
    X = data[base_features + time_features]

    poly_transformer = None
    if poly:
        poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly_transformer.fit_transform(X)
        X = pd.DataFrame(X_poly, columns=poly_transformer.get_feature_names_out(X.columns))
        numeric_features = list(X.columns)
    else:
        numeric_features = base_features + time_features

    return X, y, numeric_features, y_raw, poly_transformer

# EDA
def perform_eda(data, y_raw, target):
    print("\n🔍 EDA:")
    plt.figure(figsize=(14, 5))
    data.set_index('measured_at')[target].plot(title=f'{target} over time')
    plt.ylabel(target)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(y_raw, bins=30, kde=True)
    plt.title(f'Distribution of {target}')
    plt.grid(True)
    plt.show()

# Train Model
def train_model(X_train, y_train, numeric_features):
    scaler = StandardScaler()
    X_scaled = X_train.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])

    model = LinearRegression()
    model.fit(X_scaled, y_train)

    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)

    print("\n📌 Koefisien Model:")
    print(coef_df)

    return model, scaler, X_train.columns.tolist()

# Evaluate
def evaluate_model(model, scaler, X_test, y_test, numeric_features, feature_names, log_transform=True):
    X_scaled = X_test.copy()
    scale_cols = [f for f in numeric_features if f in feature_names]
    X_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log) if log_transform else y_pred_log
    y_true = np.expm1(y_test) if log_transform else y_test

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mean_y = y_true.mean()
    ratio = (rmse / mean_y) * 100

    print("\n📈 === HASIL EVALUASI MODEL ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Rata-rata Aktual: {mean_y:.4f}")
    print(f"Rasio RMSE terhadap Rata-rata: {ratio:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Aktual")
    plt.ylabel("Prediksi")
    plt.title("Prediksi vs Aktual")
    plt.grid(True)
    plt.show()

    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Distribusi Residual')
    plt.xlabel('Error (Aktual - Prediksi)')
    plt.grid(True)
    plt.show()

    return mse, rmse, r2

# Save Model
def save_model_components(model, scaler, features, poly_transformer, model_path, scaler_path):
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, os.path.join(os.path.dirname(model_path), 'model_features.pkl'))
        if poly_transformer:
            joblib.dump(poly_transformer, os.path.join(os.path.dirname(model_path), 'poly_transformer.pkl'))

        with open("models/last_trained.txt", "w") as f:
            f.write(f"Last trained at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"✅ Model dan komponen berhasil disimpan.")
    except Exception as e:
        print(f"❌ Gagal menyimpan: {e}")

# Retrain
def retrain_model(data_filepath, model_path, scaler_path, target='power'):
    print(f"\n🔁 Simulasi Retraining Otomatis dari {data_filepath}")
    data = load_data(data_filepath)
    X, y, numeric, y_raw, poly = prepare_data(data, target=target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    model, scaler, selected = train_model(X_train, y_train, numeric)
    evaluate_model(model, scaler, X_test, y_test, numeric, selected)
    save_model_components(model, scaler, selected, poly, model_path, scaler_path)

# MAIN
if __name__ == "__main__":
    DATA_PATH = "data/energy_measurements.csv"
    MODEL_PATH = "models/energy_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    TARGET = "power"

    os.makedirs("models", exist_ok=True)

    print("\n🔧 Pelatihan Awal Model")
    data = load_data(DATA_PATH)
    X, y, numeric, y_raw, poly = prepare_data(data, target=TARGET)
    perform_eda(data, y_raw, TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    model, scaler, selected_features = train_model(X_train, y_train, numeric)
    evaluate_model(model, scaler, X_test, y_test, numeric, selected_features)
    save_model_components(model, scaler, selected_features, poly, MODEL_PATH, SCALER_PATH)

    retrain_model(DATA_PATH, MODEL_PATH, SCALER_PATH, target=TARGET)
    print("\n✅ Proses selesai.")
