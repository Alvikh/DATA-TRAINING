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

# 1. Load Dataset
def load_data(filepath):
    """Memuat dataset dan melakukan validasi dasar"""
    try:
        data = pd.read_csv(filepath, parse_dates=['measured_at'])
        print("\nData berhasil dimuat. Contoh data:")
        print(data.head(3))
        
        # Validasi kolom yang diperlukan
        required_columns = ['measured_at', 'voltage', 'current', 'power', 'energy', 
                           'frequency', 'power_factor', 'temperature', 'humidity']
        missing_cols = [col for col in required_columns if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Kolom yang diperlukan tidak ditemukan: {missing_cols}")
            
        return data
    except Exception as e:
        print(f"\n❌ Gagal memuat data: {e}")
        exit()

# 2. Preprocess dan Feature Engineering
def prepare_data(data, target='power'):
    """Mempersiapkan data dengan feature engineering"""
    data = data.copy()
    
    # 1. Feature engineering waktu
    data['hour'] = data['measured_at'].dt.hour
    data['day_of_week'] = data['measured_at'].dt.dayofweek
    data['month'] = data['measured_at'].dt.month
    data['is_weekend'] = (data['measured_at'].dt.dayofweek >= 5).astype(int)
    
    # 2. Pilih fitur
    numeric_features = [
        'voltage', 'current', 'energy', 
        'frequency', 'power_factor', 'temperature', 'humidity'
    ]
    
    time_features = ['hour', 'day_of_week', 'month', 'is_weekend']
    
    # 3. Pastikan target tidak termasuk dalam fitur
    if target in numeric_features:
        numeric_features.remove(target)
    
    all_features = numeric_features + time_features
    
    X = data[all_features]
    y = data[target]
    
    return X, y, numeric_features, time_features

# 3. Exploratory Data Analysis
def perform_eda(data, X, y, target):
    """Analisis eksplorasi data"""
    print("\n🔍 Exploratory Data Analysis")
    
    # 1. Plot target over time
    plt.figure(figsize=(14, 5))
    data.set_index('measured_at')[target].plot(title=f'{target.capitalize()} Over Time')
    plt.ylabel(target)
    plt.grid(True)
    plt.show()
    
    # 2. Distribusi target
    plt.figure(figsize=(10, 5))
    sns.histplot(y, bins=30, kde=True)
    plt.title(f'Distribution of {target}')
    plt.show()
    
    # 3. Correlation matrix
    plt.figure(figsize=(12, 8))
    corr = pd.concat([X, y], axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

# 4. Train Model dengan Validasi
def train_model(X_train, y_train, numeric_features):
    """Melatih model dengan validasi"""
    # 1. Standardisasi fitur numerik
    scaler = StandardScaler()
    X_scaled = X_train.copy()
    X_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    # 2. Latih model
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    
    # 3. Cek koefisien
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)
    
    print("\nKoefisien Model Awal:")
    print(coef_df)
    
    # 4. Identifikasi fitur tidak penting (koefisien mendekati 0)
    insignificant_features = coef_df[abs(coef_df['Coefficient']) < 1e-5]['Feature'].tolist()
    
    if insignificant_features:
        print(f"\n⚠️ Fitur tidak signifikan terdeteksi: {insignificant_features}")
        print("Mencoba melatih ulang tanpa fitur-fitur ini...")
        
        # Filter fitur
        significant_features = [f for f in X_train.columns if f not in insignificant_features]
        X_train_filtered = X_train[significant_features]
        
        # Latih ulang model
        model_filtered = LinearRegression()
        X_scaled_filtered = X_train_filtered.copy()
        
        # Hanya standarisasi fitur numerik yang tersisa
        remaining_numeric = [f for f in numeric_features if f in significant_features]
        if remaining_numeric:
            X_scaled_filtered[remaining_numeric] = scaler.fit_transform(X_train_filtered[remaining_numeric])
        
        model_filtered.fit(X_scaled_filtered, y_train)
        
        # Evaluasi ulang koefisien
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
    
    # Hanya standarisasi fitur numerik yang digunakan dalam pelatihan
    numeric_to_scale = [f for f in numeric_features if f in feature_names]
    if numeric_to_scale:
        X_scaled[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
    
    y_pred = model.predict(X_scaled)
    
    # Hitung metrik
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 Evaluasi Model:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Plot hasil
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Nilai Aktual")
    plt.ylabel("Prediksi Model")
    plt.title("Perbandingan Nilai Aktual vs Prediksi")
    plt.grid(True)
    plt.show()
    
    # Plot residual
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Distribusi Residual')
    plt.xlabel('Error (Aktual - Prediksi)')
    plt.grid(True)
    plt.show()
    
    return rmse, r2

# Main Execution
if __name__ == "__main__":
    # Konfigurasi
    DATA_PATH = "data/energy_measurements.csv"
    MODEL_PATH = "models/energy_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    TARGET = 'power'
    
    print("=== Pemodelan Prediksi Konsumsi Energi ===")
    
    # 1. Load data
    print("\n1. Memuat data...")
    data = load_data(DATA_PATH)
    
    # 2. Persiapan data
    print("\n2. Mempersiapkan data...")
    X, y, numeric_features, time_features = prepare_data(data, target=TARGET)
    
    # 3. EDA
    print("\n3. Analisis data eksploratif...")
    perform_eda(data, X, y, TARGET)
    
    # 4. Split data
    print("\n4. Membagi data training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 5. Training model
    print("\n5. Melatih model...")
    model, scaler, selected_features = train_model(X_train, y_train, numeric_features)
    
    # 6. Evaluasi model
    print("\n6. Mengevaluasi model...")
    evaluate_model(model, scaler, X_test, y_test, numeric_features, selected_features)
    
    # 7. Simpan model
    print("\n7. Menyimpan model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Model disimpan di {MODEL_PATH}")
    print(f"✅ Scaler disimpan di {SCALER_PATH}")
    
    # 8. Tampilkan fitur penting
    coef_df = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)
    
    print("\n🎯 Fitur Penting:")
    print(coef_df)