import pandas as pd
import numpy as np # Diperlukan untuk np.nan
import os
def clean_sensor_data(filepath, columns_to_clean, output_filepath=None):
    """
    Membersihkan data sensor di file CSV/Excel dengan mengganti nilai 0 
    pada kolom tertentu dengan nilai valid terdekat di atasnya (forward fill).

    Args:
        filepath (str): Jalur ke file dataset (misal: 'data/energy_measurements.csv').
        columns_to_clean (list): Daftar nama kolom yang nilai 0-nya akan diganti.
        output_filepath (str, optional): Jalur untuk menyimpan file yang sudah dibersihkan.
                                         Jika None, data akan dikembalikan sebagai DataFrame.
    Returns:
        pandas.DataFrame or None: DataFrame yang sudah dibersihkan jika output_filepath None,
                                  jika tidak, None (data disimpan ke file).
    """
    print(f"Memuat data dari: {filepath}")
    try:
        # Coba baca sebagai CSV, jika gagal coba sebagai Excel
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, parse_dates=['measured_at'])
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath, parse_dates=['measured_at'])
        else:
            raise ValueError("Format file tidak didukung. Harap gunakan .csv, .xls, atau .xlsx.")
            
        print(f"Jumlah baris awal: {len(df)}")
        print("Contoh 5 baris pertama sebelum pembersihan:")
        print(df[columns_to_clean].head())

        for col in columns_to_clean:
            if col in df.columns:
                print(f"\nMembersihkan kolom: '{col}'...")
                
                # Hitung jumlah 0 sebelum penggantian
                zeros_count = (df[col] == 0).sum()
                print(f"Jumlah nilai 0 di '{col}' sebelum penggantian: {zeros_count}")

                # Langkah 1 & 2: Ganti 0 dengan NaN
                # Pastikan tipe data kolom adalah numerik
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(0, np.nan)
                
                # Langkah 3: Isi NaN dengan forward fill (nilai terdekat di atasnya)
                df[col] = df[col].ffill()
                
                # Handle kasus jika ada NaN di awal kolom (misal baris pertama adalah 0)
                # Jika ffill tidak bisa mengisi karena NaN ada di awal, bisa diisi dengan nilai non-zero pertama yang valid
                # Atau dengan rata-rata, median, atau nilai default lain yang masuk akal
                if df[col].isnull().sum() > 0:
                    print(f"⚠️ Peringatan: Masih ada NaN di awal kolom '{col}' setelah ffill. Mengisi dengan nilai non-NaN pertama.")
                    df[col] = df[col].bfill() # Backward fill jika masih ada NaN (misal 0 di baris pertama)
                
                # Verifikasi bahwa tidak ada lagi 0 atau NaN yang tersisa di kolom yang dibersihkan
                zeros_after = (df[col] == 0).sum()
                nan_after = df[col].isnull().sum()
                print(f"Jumlah nilai 0 di '{col}' setelah penggantian: {zeros_after}")
                print(f"Jumlah nilai NaN di '{col}' setelah penggantian: {nan_after}")
                
            else:
                print(f"Kolom '{col}' tidak ditemukan di dataset. Melewati.")

        print("\nContoh 5 baris pertama setelah pembersihan:")
        print(df[columns_to_clean].head())
        print(f"Jumlah baris akhir: {len(df)}")
        
        if output_filepath:
            # Pastikan direktori output ada
            output_dir = os.path.dirname(output_filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if output_filepath.endswith('.csv'):
                df.to_csv(output_filepath, index=False)
            elif output_filepath.endswith(('.xls', '.xlsx')):
                df.to_excel(output_filepath, index=False)
            print(f"\nData yang sudah dibersihkan disimpan ke: {output_filepath}")
            return None
        else:
            print("\nData yang sudah dibersihkan dikembalikan sebagai DataFrame.")
            return df

    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di '{filepath}'")
        return None
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat membersihkan data: {e}")
        return None

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    # Path ke file Excel Anda
    input_file = 'data/energy_measurements.csv' # Ganti dengan nama file Excel Anda
    
    # Path untuk menyimpan file yang sudah dibersihkan
    # Disarankan disimpan sebagai CSV untuk proses ML selanjutnya
    output_file = 'data/energy_measurements_cleaned.csv' 
    
    # Kolom yang ingin Anda bersihkan dari nilai 0 yang tidak masuk akal
    columns_to_process = ['temperature', 'humidity', 'voltage', 'current', 'power', 'energy', 'frequency', 'power_factor'] 
    # Anda bisa tambahkan atau kurangi kolom sesuai kebutuhan.
    # Misalnya, voltage, current, power, energy, frequency, power_factor juga bisa punya 0 yang berarti "tidak ada daya".
    # Pertimbangkan apakah 0 di kolom ini harus diganti atau diartikan sebagai "tidak ada aktivitas".
    # Jika 0 berarti tidak ada aktivitas (misal daya), maka jangan diganti.
    # Jika 0 adalah indikator sensor error, maka ganti.
    
    # Jalankan fungsi pembersihan
    cleaned_df = clean_sensor_data(input_file, columns_to_process, output_filepath=output_file)
    
    # Jika Anda ingin bekerja dengan DataFrame secara langsung tanpa menyimpannya ke file,
    # panggil fungsi tanpa argumen output_filepath:
    # cleaned_df_in_memory = clean_sensor_data(input_file, columns_to_process)
    # if cleaned_df_in_memory is not None:
    #     print("\nDataFrame yang sudah dibersihkan (dalam memori):")
    #     print(cleaned_df_in_memory.head())