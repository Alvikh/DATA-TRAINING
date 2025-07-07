from .db import MySQLDatabase
from datetime import datetime
from typing import List, Dict, Optional

class EnergyMeasurement:
    """
    CRUD operations for energy_measurements table with the following structure:
    
    CREATE TABLE IF NOT EXISTS energy_measurements (
        id INT AUTO_INCREMENT PRIMARY KEY,
        device_id VARCHAR(50) NOT NULL,
        voltage DECIMAL(10,2) NOT NULL,
        current DECIMAL(10,2) NOT NULL,
        power DECIMAL(10,2) NOT NULL,
        energy DECIMAL(10,2) NOT NULL,
        frequency DECIMAL(10,2),
        power_factor DECIMAL(10,2),
        temperature DECIMAL(10,2),
        humidity DECIMAL(10,2),
        measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX (device_id),
        INDEX (measured_at)
    )
    """
    
    def __init__(self, db_config: dict):
        """
        Initialize with database configuration
        
        Args:
            db_config: Dictionary containing host, user, password, database
        """
        self.db_config = db_config

    def create(self, measurement_data: Dict) -> Optional[int]:
        """
        Create a new energy measurement record
        
        Args:
            measurement_data: Dictionary containing measurement values
                Required keys: device_id, voltage, current, power, energy
                Optional keys: frequency, power_factor, temperature, humidity, measured_at
                
        Returns:
            ID of the newly created record or None if failed
        """
        required_fields = ['device_id', 'voltage', 'current', 'power', 'energy']
        if not all(field in measurement_data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        query = """
            INSERT INTO energy_measurements 
            (device_id, voltage, current, power, energy, frequency, 
            power_factor, temperature, humidity, measured_at,
            created_at, updated_at)
            VALUES (%(device_id)s, %(voltage)s, %(current)s, %(power)s, %(energy)s, 
                    %(frequency)s, %(power_factor)s, %(temperature)s, %(humidity)s, 
                    %(measured_at)s,
                    NOW(), NOW())
            """
        
        # Set default values for optional fields
        measurement_data.setdefault('frequency', None)
        measurement_data.setdefault('power_factor', None)
        measurement_data.setdefault('temperature', None)
        measurement_data.setdefault('humidity', None)
        measurement_data.setdefault('measured_at', datetime.now())
        
        with MySQLDatabase(**self.db_config) as db:
            if db.execute_query(query, measurement_data):
                return db.connection.cursor().lastrowid
            return None

    def get_by_id(self, measurement_id: int) -> Optional[Dict]:
        """
        Get a single measurement by ID
        
        Args:
            measurement_id: ID of the measurement record
            
        Returns:
            Dictionary with measurement data or None if not found
        """
        query = "SELECT * FROM energy_measurements WHERE id = %s"
        with MySQLDatabase(**self.db_config) as db:
            results = db.execute_query(query, (measurement_id,), fetch=True)
            return results[0] if results else None

    def get_by_device(
        self, 
        device_id: str, 
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get measurements for a specific device
        
        Args:
            device_id: Device identifier
            limit: Maximum number of records to return
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of measurement dictionaries
        """
        query = """
        SELECT * FROM energy_measurements 
        WHERE device_id = %s
        """
        params = [device_id]
        
        if start_date:
            query += " AND measured_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND measured_at <= %s"
            params.append(end_date)
            
        query += " ORDER BY measured_at DESC LIMIT %s"
        params.append(limit)
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params), fetch=True) or []

    def update(self, measurement_id: int, update_data: Dict) -> bool:
        """
        Update an existing measurement record
        
        Args:
            measurement_id: ID of the record to update
            update_data: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not update_data:
            raise ValueError("No update data provided")
            
        set_clauses = []
        params = []
        
        for field, value in update_data.items():
            set_clauses.append(f"{field} = %s")
            params.append(value)
            
        params.append(measurement_id)
        
        query = f"""
        UPDATE energy_measurements 
        SET {', '.join(set_clauses)}
        WHERE id = %s
        """
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params))

    def delete(self, measurement_id: int) -> bool:
        """
        Delete a measurement record
        
        Args:
            measurement_id: ID of the record to delete
            
        Returns:
            True if successful, False otherwise
        """
        query = "DELETE FROM energy_measurements WHERE id = %s"
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, (measurement_id,))

    def get_aggregated_stats(
        self,
        device_id: str,
        time_window: str = 'hour',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get aggregated statistics for a device
        
        Args:
            device_id: Device identifier
            time_window: Aggregation window ('hour', 'day', 'week', 'month')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of aggregated statistics
        """
        valid_windows = ['hour', 'day', 'week', 'month']
        if time_window not in valid_windows:
            raise ValueError(f"Invalid time window. Must be one of: {valid_windows}")
            
        query = f"""
        SELECT 
            DATE_FORMAT(measured_at, %s) as time_period,
            AVG(voltage) as avg_voltage,
            AVG(current) as avg_current,
            SUM(power) as total_power,
            SUM(energy) as total_energy,
            AVG(frequency) as avg_frequency,
            AVG(power_factor) as avg_power_factor,
            AVG(temperature) as avg_temperature,
            AVG(humidity) as avg_humidity
        FROM energy_measurements
        WHERE device_id = %s
        """
        
        params = []
        
        # Determine date format based on time window
        if time_window == 'hour':
            date_format = '%Y-%m-%d %H:00:00'
        elif time_window == 'day':
            date_format = '%Y-%m-%d'
        elif time_window == 'week':
            date_format = '%Y-%u'
        else:  # month
            date_format = '%Y-%m'
            
        params.append(date_format)
        params.append(device_id)
        
        if start_date:
            query += " AND measured_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND measured_at <= %s"
            params.append(end_date)
            
        query += f" GROUP BY time_period ORDER BY time_period"
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params), fetch=True) or []


# Example usage
if __name__ == "__main__":
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'admin',
        'password': 'admin',
        'database': 'powersmart'
    }
    
    # Initialize EnergyMeasurement
    energy_meter = EnergyMeasurement(DB_CONFIG)
    
    # 1. Create a new measurement
    new_measurement = {
        'device_id': 'PS-1001',
        'voltage': 220.5,
        'current': 1.23,
        'power': 271.22,
        'energy': 0.15,
        'frequency': 50.1,
        'power_factor': 0.98,
        'temperature': 28.5,
        'humidity': 65.2,
        'measured_at': datetime.now()
    }
    
    measurement_id = energy_meter.create(new_measurement)
    print(f"Created measurement with ID: {measurement_id}")
    
    # 2. Get measurement by ID
    measurement = energy_meter.get_by_id(measurement_id)
    print(f"\nMeasurement details:\n{measurement}")
    
    # 3. Get measurements for device
    device_measurements = energy_meter.get_by_device('PS-1001', limit=5)
    print(f"\nLast 5 measurements for device:")
    for m in device_measurements:
        print(f"{m['measured_at']} - {m['power']}W")
    
    # 4. Update measurement
    update_success = energy_meter.update(measurement_id, {'power': 275.0})
    print(f"\nUpdate successful: {update_success}")
    
    # 5. Get aggregated stats
    stats = energy_meter.get_aggregated_stats('PS-1001', time_window='hour')
    print("\nHourly aggregated stats:")
    for stat in stats:
        print(f"{stat['time_period']}: {stat['total_power']:.2f}W")
    
    # 6. Delete measurement
    delete_success = energy_meter.delete(measurement_id)
    print(f"\nDelete successful: {delete_success}")