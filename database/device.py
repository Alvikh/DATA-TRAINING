from .db import MySQLDatabase
from datetime import datetime
from typing import List, Dict, Optional

class DeviceManager:
    """
    CRUD operations for device management with the following structure:
    
    CREATE TABLE IF NOT EXISTS devices (
        id INT AUTO_INCREMENT PRIMARY KEY,
        owner_id VARCHAR(50) NOT NULL,
        name VARCHAR(100) NOT NULL,
        device_id VARCHAR(50) NOT NULL UNIQUE,
        type VARCHAR(50) NOT NULL,
        building VARCHAR(100),
        installation_date DATE,
        status VARCHAR(20) DEFAULT 'active',
        state VARCHAR(20) DEFAULT 'offline',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX (owner_id),
        INDEX (device_id),
        INDEX (status),
        INDEX (state)
    )
    """
    
    def __init__(self, db_config: dict):
        self.db_config = db_config

    def create_device(self, device_data: Dict) -> Optional[int]:
        """
        Create a new device record
        
        Args:
            device_data: Dictionary containing device values
                Required keys: owner_id, name, device_id, type
                Optional keys: building, installation_date, status, state
                
        Returns:
            ID of the newly created device or None if failed
        """
        required_fields = ['owner_id', 'name', 'device_id', 'type']
        if not all(field in device_data for field in required_fields):
            raise ValueError(f"Missing required fields: {required_fields}")

        query = """
        INSERT INTO devices 
        (owner_id, name, device_id, type, building, 
         installation_date, status, state)
        VALUES (%(owner_id)s, %(name)s, %(device_id)s, %(type)s, 
                %(building)s, %(installation_date)s, %(status)s, %(state)s)
        """
        
        # Set default values
        device_data.setdefault('status', 'active')
        device_data.setdefault('state', 'offline')
        device_data.setdefault('building', None)
        device_data.setdefault('installation_date', None)
        
        with MySQLDatabase(**self.db_config) as db:
            if db.execute_query(query, device_data):
                return db.connection.cursor().lastrowid
            return None
    def device_exists(self, device_id: str) -> bool:
        """Check if a device exists in the database"""
        query = "SELECT 1 FROM devices WHERE device_id = %s LIMIT 1"
        with MySQLDatabase(**self.db_config) as db:
            result = db.execute_query(query, (device_id,), fetch=True)
            return bool(result)
    def get_device(self, device_id: str) -> Optional[Dict]:
        """
        Get device by device_id (not the primary key ID)
        
        Args:
            device_id: The unique device identifier
            
        Returns:
            Dictionary with device data or None if not found
        """
        query = "SELECT * FROM devices WHERE device_id = %s"
        with MySQLDatabase(**self.db_config) as db:
            results = db.execute_query(query, (device_id,), fetch=True)
            return results[0] if results else None

    def get_devices_by_owner(
        self, 
        owner_id: str,
        status: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get devices for a specific owner with optional filters
        
        Args:
            owner_id: Owner identifier
            status: Optional status filter (e.g., 'active', 'inactive')
            state: Optional state filter (e.g., 'online', 'offline')
            limit: Maximum number of records to return
            
        Returns:
            List of device dictionaries
        """
        query = "SELECT * FROM devices WHERE owner_id = %s"
        params = [owner_id]
        
        if status:
            query += " AND status = %s"
            params.append(status)
        if state:
            query += " AND state = %s"
            params.append(state)
            
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params), fetch=True) or []

    def update_device(self, device_id: str, update_data: Dict) -> bool:
        """
        Update device information
        
        Args:
            device_id: The unique device identifier (not primary key ID)
            update_data: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not update_data:
            raise ValueError("No update data provided")
            
        # Prevent updating certain fields
        restricted_fields = {'id', 'created_at'}
        for field in restricted_fields:
            if field in update_data:
                raise ValueError(f"Cannot update restricted field: {field}")
                
        set_clauses = []
        params = []
        
        for field, value in update_data.items():
            set_clauses.append(f"{field} = %s")
            params.append(value)
            
        params.append(device_id)
        
        query = f"""
        UPDATE devices 
        SET {', '.join(set_clauses)}
        WHERE device_id = %s
        """
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params))

    def update_device_state(self, device_id: str, new_state: str) -> bool:
        """
        Update the state of a device (e.g., turn ON/OFF)

        Args:
            device_id: The unique device identifier
            new_state: New state value ('ON' or 'OFF')

        Returns:
            True if successful, False otherwise
        """
        valid_states = {'ON', 'OFF'}
        if new_state.upper() not in valid_states:
            raise ValueError(f"Invalid state. Must be one of: {valid_states} your state was {new_state}")
        
        query = """
        UPDATE devices 
        SET state = %s 
        WHERE device_id = %s
        """
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, (new_state.lower(), device_id))

    def delete_device(self, device_id: str) -> bool:
        """
        Delete a device record
        
        Args:
            device_id: The unique device identifier
            
        Returns:
            True if successful, False otherwise
        """
        query = "DELETE FROM devices WHERE device_id = %s"
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, (device_id,))

    def get_devices_by_state(self, state: str) -> List[Dict]:
        """
        Get all devices with a specific state
        
        Args:
            state: Device state to filter by
            
        Returns:
            List of devices in the specified state
        """
        query = "SELECT * FROM devices WHERE state = %s ORDER BY updated_at DESC"
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, (state.lower(),), fetch=True) or []

    def bulk_update_state(self, device_ids: List[str], new_state: str) -> int:
        """
        Update state for multiple devices at once
        
        Args:
            device_ids: List of device identifiers
            new_state: New state value
            
        Returns:
            Number of devices successfully updated
        """
        if not device_ids:
            return 0
            
        valid_states = {'online', 'offline', 'maintenance', 'error'}
        if new_state.lower() not in valid_states:
            raise ValueError(f"Invalid state. Must be one of: {valid_states}")
        
        # Create a tuple of parameters (new_state followed by device_ids)
        params = [new_state.lower()] + device_ids
        
        # Create placeholders for the IN clause
        placeholders = ', '.join(['%s'] * len(device_ids))
        
        query = f"""
        UPDATE devices 
        SET state = %s 
        WHERE device_id IN ({placeholders})
        """
        
        with MySQLDatabase(**self.db_config) as db:
            if db.execute_query(query, tuple(params)):
                return db.connection.cursor().rowcount
            return 0