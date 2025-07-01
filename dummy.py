import pandas as pd
import numpy as np

# Set the number of rows
num_rows = 1000000

# Generate dummy data for each feature
np.random.seed(42) # for reproducibility

# Tambahkan kolom Timestamp
# Mulai dari waktu tertentu (misal: 1 Januari 2023, 00:00:00)
start_time = pd.Timestamp('2023-01-01 00:00:00')
timestamps = pd.date_range(start=start_time, periods=num_rows, freq='min')

voltage = np.random.uniform(210, 240, num_rows)
current = np.random.uniform(0.1, 10, num_rows)
power = np.random.uniform(20, 2000, num_rows)
energy = np.random.uniform(0.01, 5, num_rows)
frequency = np.random.uniform(49.5, 50.5, num_rows)
power_factor = np.random.uniform(0.5, 1.0, num_rows)
temperature = np.random.uniform(20, 35, num_rows)
humidity = np.random.uniform(30, 80, num_rows)

# Generate Output Energy with fluctuation
# A simple way to add fluctuation: energy + a small random value (positive or negative)
output_energy = energy + np.random.uniform(-0.1, 0.1, num_rows)
# Ensure output_energy doesn't go below 0 (though unlikely with current range)
output_energy = np.maximum(output_energy, 0.01)

# Create a DataFrame with the desired column names
data = pd.DataFrame({
    'timestamp': timestamps,
    'voltage': voltage,
    'current': current,
    'power': power,
    'energy': energy,
    'frequency': frequency,
    'power_factor': power_factor,
    'temperature': temperature,
    'humidity': humidity,
    'output_energy': output_energy
})

# Save to CSV
file_path = 'data/energy_data.csv' # Nama file baru
data.to_csv(file_path, index=False)

print(f"Dataset dummy dengan format kolom yang diminta berhasil dibuat dan disimpan di: {file_path}")
print(f"Jumlah baris dalam dataset: {len(data)}")
print("\nContoh 5 baris pertama dari dataset:")
print(data.head())