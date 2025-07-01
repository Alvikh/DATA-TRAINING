import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. Fungsi untuk memuat data
def load_data(filepath):
    """Memuat dataset dari file CSV dan melakukan preprocessing waktu"""
    try:
        # Membaca data dengan parsing kolom waktu
        data = pd.read_csv(filepath, parse_dates=['timestamp'])  # Ganti 'timestamp' sesuai nama kolom di dataset
        
        print("Data berhasil dimuat. 5 data pertama:")
        print(data.head())
        print("\nInformasi dataset:")
        print(data.info())
        
        return data
    
    except FileNotFoundError:
        print("Error: File tidak ditemukan. Pastikan path file benar.")
        exit()
    except KeyError:
        print("Error: Kolom 'timestamp' tidak ditemukan. Pastikan dataset memiliki kolom waktu.")
        exit()

# 2. Fungsi untuk mempersiapkan data dengan fitur waktu
def prepare_data(data):
    """Memisahkan fitur (X) dan target (y) dengan ekstraksi fitur waktu"""
    # Fitur dasar
    base_features = ['voltage', 'current', 'power', 'energy', 
                    'frequency', 'power_factor', 'temperature', 'humidity']
    
    # Target
    target = 'output_energy'
    
    # Memeriksa ketersediaan kolom
    missing_features = [f for f in base_features if f not in data.columns]
    if missing_features:
        print(f"Peringatan: Kolom berikut tidak ditemukan: {missing_features}")
        base_features = [f for f in base_features if f in data.columns]
    
    # Ekstraksi fitur waktu
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek  # 0=Senin, 6=Minggu
    data['month'] = data['timestamp'].dt.month
    data['is_weekend'] = (data['timestamp'].dt.dayofweek >= 5).astype(int)
    
    # Gabungkan semua fitur
    all_features = base_features + ['hour', 'day_of_week', 'month', 'is_weekend']
    
    X = data[all_features]
    y = data[target]
    
    print("\nFitur yang digunakan:")
    print(X.columns.tolist())
    
    return X, y

# 3. Fungsi untuk analisis data eksploratif (EDA)
def perform_eda(data, X, y):
    """Melakukan analisis data eksploratif dengan visualisasi"""
    print("\nMemulai Analisis Data Eksploratif...")
    
    # 3.1. Tren waktu konsumsi energi
    plt.figure(figsize=(14, 6))
    data.set_index('timestamp')['output_energy'].plot(title='Tren Konsumsi Energi Over Time')
    plt.ylabel('Konsumsi Energi')
    plt.grid(True)
    plt.show()
    
    # 3.2. Matriks korelasi
    plt.figure(figsize=(12, 8))
    corr_matrix = pd.concat([X, y], axis=1).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriks Korelasi Fitur')
    plt.tight_layout()
    plt.show()
    
    # 3.3. Distribusi fitur numerik
    X.hist(figsize=(14, 10), bins=20)
    plt.suptitle('Distribusi Fitur Numerik', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # 3.4. Visualisasi hubungan waktu dengan konsumsi energi
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='hour', y='output_energy', data=pd.concat([X, y], axis=1))
    plt.title('Konsumsi Energi per Jam')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='day_of_week', y='output_energy', data=pd.concat([X, y], axis=1))
    plt.title('Konsumsi Energi per Hari dalam Minggu')
    
    plt.tight_layout()
    plt.show()

# 4. Fungsi untuk melatih model
def train_model(X_train, y_train):
    """Melatih model regresi linear dengan standardisasi"""
    # Standardisasi fitur (kecuali variabel waktu yang sudah dalam skala tepat)
    numeric_features = ['voltage', 'current', 'power', 'energy', 
                       'frequency', 'temperature', 'humidity']
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    # Melatih model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# 5. Fungsi untuk evaluasi model
def evaluate_model(model, scaler, X_test, y_test):
    """Mengevaluasi performa model"""
    # Standardisasi data test
    X_test_scaled = X_test.copy()
    numeric_features = ['voltage', 'current', 'power', 'energy', 
                       'frequency', 'temperature', 'humidity']
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    
    # Metrik evaluasi
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nHasil Evaluasi Model:")
    print(f"- MSE: {mse:.4f}")
    print(f"- R-squared: {r2:.4f}")
    
    # Visualisasi prediksi vs aktual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Nilai Aktual')
    plt.ylabel('Prediksi')
    plt.title('Perbandingan Nilai Aktual vs Prediksi')
    plt.grid(True)
    plt.show()
    
    return mse, r2

# 6. Fungsi untuk menampilkan koefisien model
def show_coefficients(model, feature_names):
    """Menampilkan koefisien model secara terurut"""
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    
    print("\nKoefisien Model (diurutkan berdasarkan pengaruh):")
    print(coef_df)
    print(f"\nIntercept: {model.intercept_:.4f}")

# Main execution
if __name__ == "__main__":
    # Konfigurasi
    DATA_PATH = "data/energy_data.csv"  # Update dengan path dataset Anda
    MODEL_PATH = "models/energy_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    
    # 1. Memuat data
    print("\n=== Memuat Data ===")
    data = load_data(DATA_PATH)
    
    # 2. Memisahkan fitur dan target
    print("\n=== Mempersiapkan Data ===")
    X, y = prepare_data(data)
    
    # 3. Analisis data eksploratif
    print("\n=== Analisis Data Eksploratif ===")
    perform_eda(data, X, y)
    
    # 4. Membagi data menjadi training dan testing set
    print("\n=== Membagi Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Ukuran Data:")
    print(f"- Training: {X_train.shape[0]} sampel")
    print(f"- Testing: {X_test.shape[0]} sampel")
    
    # 5. Melatih model
    print("\n=== Melatih Model ===")
    model, scaler = train_model(X_train, y_train)
    
    # 6. Evaluasi model
    print("\n=== Evaluasi Model ===")
    mse, r2 = evaluate_model(model, scaler, X_test, y_test)
    
    # 7. Menampilkan koefisien model
    show_coefficients(model, X.columns.tolist())
    
    # 8. Menyimpan model dan scaler
    print("\n=== Menyimpan Model ===")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model disimpan ke {MODEL_PATH}")
    print(f"Scaler disimpan ke {SCALER_PATH}")
    
    # 9. Saran untuk peningkatan
    print("\n=== Saran untuk Peningkatan ===")
    print("1. Coba algoritma lain seperti Random Forest atau XGBoost untuk menangani non-linearitas")
    print("2. Tambahkan fitur interaksi (misalnya voltage*current)")
    print("3. Eksperimen dengan transformasi fitur (log, polynomial)")
    print("4. Gunakan cross-validation untuk evaluasi yang lebih robust")
    print("5. Pertimbangkan model time-series khusus seperti ARIMA jika pola temporal kuat")