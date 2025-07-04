import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Tetap pakai LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os # Tambahkan import os untuk path

# --- (FUNGSI YANG SUDAH ADA SEBELUMNYA) ---
# 1. Load Dataset
def load_data(filepath):
    """Memuat dataset dan melakukan validasi dasar"""
    try:
        data = pd.read_csv(filepath, parse_dates=['measured_at'])
        print(f"\nData dari {filepath} berhasil dimuat. Contoh data:")
        print(data.head(3))
        
        required_columns = ['measured_at', 'voltage', 'current', 'power', 'energy', 
                            'frequency', 'power_factor', 'temperature', 'humidity']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {missing_cols}")
            
        return data
    except Exception as e:
        print(f"\n❌ Gagal memuat data dari {filepath}: {e}")
        exit()

# 2. Preprocess dan Feature Engineering
def prepare_data(data, target='power'):
    """Mempersiapkan data dengan feature engineering"""
    data = data.copy()
    
    data['hour'] = data['measured_at'].dt.hour
    data['day_of_week'] = data['measured_at'].dt.dayofweek
    data['month'] = data['measured_at'].dt.month
    data['is_weekend'] = (data['measured_at'].dt.dayofweek >= 5).astype(int)
    
    numeric_features = [
        'voltage', 'current', 'energy', 
        'frequency', 'power_factor', 'temperature', 'humidity'
    ]
    
    time_features = ['hour', 'day_of_week', 'month', 'is_weekend']
    
    if target in numeric_features:
        numeric_features.remove(target)
    
    all_features = numeric_features + time_features
    
    X = data[all_features]
    y = data[target]
    
    return X, y, numeric_features, time_features

# 3. Exploratory Data Analysis (opsional, bisa dilewati saat retraining otomatis)
def perform_eda(data, X, y, target):
    """Analisis eksplorasi data"""
    print("\n🔍 Exploratory Data Analysis")
    
    plt.figure(figsize=(14, 5))
    data.set_index('measured_at')[target].plot(title=f'{target.capitalize()} Over Time')
    plt.ylabel(target)
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.histplot(y, bins=30, kde=True)
    plt.title(f'Distribution of {target}')
    plt.show()
    
    plt.figure(figsize=(12, 8))
    corr = pd.concat([X, y], axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

# 4. Train Model dengan Validasi
def train_model(X_train, y_train, numeric_features):
    """Melatih model dengan validasi"""
    scaler = StandardScaler()
    X_scaled = X_train.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)
    
    print("\nKoefisien Model Awal:")
    print(coef_df)
    
    insignificant_features = coef_df[abs(coef_df['Coefficient']) < 1e-5]['Feature'].tolist()
    
    if insignificant_features:
        print(f"\n⚠️ Fitur tidak signifikan terdeteksi: {insignificant_features}")
        print("Mencoba melatih ulang tanpa fitur-fitur ini...")
        
        significant_features = [f for f in X_train.columns if f not in insignificant_features]
        X_train_filtered = X_train[significant_features]
        
        model_filtered = LinearRegression()
        X_scaled_filtered = X_train_filtered.copy()
        
        remaining_numeric = [f for f in numeric_features if f in significant_features]
        if remaining_numeric:
            X_scaled_filtered[remaining_numeric] = scaler.fit_transform(X_train_filtered[remaining_numeric])
        
        model_filtered.fit(X_scaled_filtered, y_train)
        
        coef_filtered = pd.DataFrame({
            "Feature": significant_features,
            "Coefficient": model_filtered.coef_
        }).sort_values("Coefficient", ascending=False)
        
        print("\nKoefisien Model setelah Filter Fitur:")
        print(coef_filtered)
        
        return model_filtered, scaler, significant_features
    
    return model, scaler, X_train.columns.tolist()

# 5. Evaluasi Model
def evaluate_model(model, scaler, X_test, y_test, numeric_features, feature_names):
    """Evaluasi performa model"""
    X_scaled = X_test.copy()
    
    numeric_to_scale = [f for f in numeric_features if f in feature_names]
    if numeric_to_scale:
        X_scaled[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
    
    y_pred = model.predict(X_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Evaluasi Model:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Nilai Aktual")
    plt.ylabel("Prediksi Model")
    plt.title("Perbandingan Nilai Aktual vs Prediksi")
    plt.grid(True)
    plt.show()
    
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Distribusi Residual')
    plt.xlabel('Error (Aktual - Prediksi)')
    plt.grid(True)
    plt.show()
    
    return rmse, r2


def retrain_model(
    data_filepath: str,
    model_path: str,
    scaler_path: str,
    target_column: str = 'power',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Melatih ulang model regresi linear dengan dataset terbaru dan menyimpannya.

    Args:
        data_filepath (str): Path ke file CSV dataset yang diperbarui.
        model_path (str): Path untuk menyimpan model LinearRegression yang baru dilatih.
        scaler_path (str): Path untuk menyimpan StandardScaler yang baru dilatih.
        target_column (str): Nama kolom target untuk prediksi (default: 'power').
        test_size (float): Proporsi dataset yang akan digunakan untuk pengujian (default: 0.2).
        random_state (int): Seed untuk pengacakan split data (default: 42).
    """
    print(f"\n=== Memulai Retraining Model dengan data dari: {data_filepath} ===")
    
    # 1. Muat seluruh dataset yang diperbarui
    print("1. Memuat seluruh data...")
    data = load_data(data_filepath)
    if data is None:
        print("❌ Retraining dibatalkan karena gagal memuat data.")
        return

    # 2. Persiapan data dan feature engineering
    print("2. Mempersiapkan data untuk pelatihan ulang...")
    X, y, numeric_features, time_features = prepare_data(data, target=target_column)
    
    # 3. Bagi data menjadi training dan testing
    print("3. Membagi data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False # shuffle=False jika data time-series
    )
    
    # 4. Latih model baru
    print("4. Melatih model baru...")
    new_model, new_scaler, new_selected_features = train_model(X_train, y_train, numeric_features)
    
    # 5. Evaluasi model yang baru dilatih
    print("5. Mengevaluasi model yang baru dilatih...")
    rmse, r2 = evaluate_model(new_model, new_scaler, X_test, y_test, numeric_features, new_selected_features)
    
    # 6. Simpan model dan scaler yang baru dilatih
    print("6. Menyimpan model dan scaler yang diperbarui...")
    try:
        joblib.dump(new_model, model_path)
        joblib.dump(new_scaler, scaler_path)
        # Penting: Jika fitur bisa berubah setelah seleksi, simpan juga nama fiturnya
        joblib.dump(new_selected_features, os.path.join(os.path.dirname(model_path), 'model_features.pkl'))
        print(f"✅ Model baru disimpan di {model_path}")
        print(f"✅ Scaler baru disimpan di {scaler_path}")
        print(f"✅ Nama fitur model baru disimpan di {os.path.join(os.path.dirname(model_path), 'model_features.pkl')}")
        print(f"=== Retraining Model Selesai. RMSE: {rmse:.4f}, R2: {r2:.4f} ===")
    except Exception as e:
        print(f"❌ Gagal menyimpan model atau scaler: {e}")

# --- MAIN EXECUTION (diperbarui untuk menunjukkan cara memanggil fungsi retraining) ---
if __name__ == "__main__":
    # Konfigurasi
    DATA_PATH = "data/train.csv"
    MODEL_PATH = "models/energy_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    TARGET = 'power'
    
    # Pastikan folder models ada
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- Skenario 1: Pelatihan Awal (atau Retraining Manual) ---
    print("\n--- SKENARIO 1: Pelatihan Awal / Retraining Manual ---")
    # Ini akan menjalankan seluruh proses seperti sebelumnya
    # Biasanya, Anda akan menjalankan ini pertama kali, atau secara manual
    # ketika ada kebutuhan retraining mendesak.
    
    data_initial = load_data(DATA_PATH)
    if data_initial is not None:
        X, y, numeric_features, time_features = prepare_data(data_initial, target=TARGET)
        perform_eda(data_initial, X, y, TARGET) # EDA hanya saat pelatihan awal/debugging
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        model, scaler, selected_features = train_model(X_train, y_train, numeric_features)
        evaluate_model(model, scaler, X_test, y_test, numeric_features, selected_features)
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(selected_features, os.path.join('models', 'model_features.pkl')) # Simpan nama fitur
        print(f"✅ Model awal disimpan di {MODEL_PATH}")
        print(f"✅ Scaler awal disimpan di {SCALER_PATH}")
        print(f"✅ Nama fitur model disimpan di models/model_features.pkl")
    else:
        print("Tidak dapat melakukan pelatihan awal karena data tidak dimuat.")

    # --- Skenario 2: Retraining Otomatis (Simulasi) ---
    # Bayangkan ini dijalankan secara terjadwal, misalnya, setiap malam.
    # Anda perlu memastikan file `data/energy_measurements.csv` telah diperbarui
    # dengan data MQTT terbaru sebelum fungsi ini dipanggil.
    print("\n--- SKENARIO 2: Simulasi Retraining Otomatis ---")
    print("Mulai retraining model dengan data yang diperbarui...")

    retrain_model(
        data_filepath=DATA_PATH,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        target_column=TARGET
    )

    print("\n=== Proses Selesai ===")