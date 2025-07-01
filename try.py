from database.db import MySQLDatabase
from datetime import datetime
import random

# 1. Inisialisasi koneksi
DB_CONFIG = {
    'host': 'localhost',
    'user': 'admin',
    'password': 'admin',
    'database': 'powersmart'
}

def contoh_penggunaan():
    # 2. Menggunakan context manager (rekomendasi)
    with MySQLDatabase(**DB_CONFIG) as db:
        if not db.connection:
            print("Gagal terhubung ke database!")
            return

        # 3. Contoh operasi CRUD
        # Insert data dummy
        device_id = "PS-1001"
        for i in range(3):
            voltage = round(220 + random.uniform(-5, 5), 2)
            current = round(random.uniform(1, 5), 2)
            power = round(voltage * current, 2)
            
            if db.insert_measurement(device_id, voltage, current, power):
                print(f"Data terukur dimasukkan: {power}W")
            else:
                print("Gagal memasukkan data!")

        # 4. Ambil data terbaru
        print("\n10 Pembacaan Terakhir:")
        measurements = db.get_latest_measurements()
        for idx, data in enumerate(measurements, 1):
            print(f"{idx}. {data['device_id']} | {data['voltage']}V | {data['current']}A | {data['power']}W | {data['measured_at']}")

        # 5. Update data
        update_query = """
        UPDATE energy_measurements
        SET power = %s
        WHERE device_id = %s AND power < %s
        """
        if db.execute_query(update_query, (5000, device_id, 1000)):
            print(f"\nUpdate data untuk {device_id} berhasil!")

        # 6. Complex query dengan join (contoh)
        complex_query = """
        SELECT device_id, AVG(power) as avg_power
        FROM energy_measurements
        WHERE measured_at > DATE_SUB(NOW(), INTERVAL 1 DAY)
        GROUP BY device_id
        """
        results = db.execute_query(complex_query, fetch=True)
        print("\nRata-rata daya perangkat (24 jam terakhir):")
        for row in results:
            print(f"{row['device_id']}: {row['avg_power']:.2f}W")

if __name__ == "__main__":
    contoh_penggunaan()