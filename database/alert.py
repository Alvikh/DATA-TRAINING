from datetime import datetime
from typing import List, Dict, Optional, Union
import logging
from .db import MySQLDatabase

logger = logging.getLogger(__name__)

class AlertManager:
    """
    Comprehensive alert management system with CRUD operations
    
    Attributes:
        db_config (dict): Database connection configuration
        table_name (str): Name of the alerts table (default: 'alerts')
    """
    
    def __init__(self, db_config: dict, table_name: str = 'alerts'):
        """
        Initialize AlertManager with database configuration
        
        Args:
            db_config: Dictionary containing database connection parameters
                Required keys: host, user, password, database
                Optional key: port (default: 3306)
            table_name: Name of the alerts table (default: 'alerts')
        """
        self.db_config = db_config
        self.table_name = table_name
        self._verify_table_structure()

    def _verify_table_structure(self):
        """Verify the alerts table exists and has the correct structure"""
        with MySQLDatabase(**self.db_config) as db:
            if not db.table_exists(self.table_name):
                logger.warning(f"Table '{self.table_name}' does not exist")
                # Consider creating the table automatically if needed
                # self._create_table(db)

    def _create_table(self, db: MySQLDatabase):
        """Create the alerts table if it doesn't exist"""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_id VARCHAR(50) NOT NULL,
            type VARCHAR(50) NOT NULL,
            message TEXT NOT NULL,
            severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
            is_resolved BOOLEAN DEFAULT FALSE,
            resolved_at TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX (device_id),
            INDEX (is_resolved),
            INDEX (severity)
        )
        """
        return db.execute_query(query)

    def create_alert(
            self,
            device_id: int,
            alert_type: str,
            message: str,
            severity: str,
            is_resolved: bool = False,
            resolved_at: Optional[datetime] = None
        ) -> Optional[int]:
        """
        Create a new alert record

        Args:
            device_id: ID of the associated device
            alert_type: Type/category of the alert
            message: Detailed alert message
            severity: Severity level ('low', 'medium', 'high', 'critical')
            is_resolved: Whether alert is resolved (default: False)
            resolved_at: When alert was resolved (optional)

        Returns:
            ID of the created alert or None if failed
        """
        query = f"""
        INSERT INTO {self.table_name} 
        (device_id, type, message, severity, is_resolved, resolved_at, created_at,updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        now = datetime.now()
        params = (
            str(device_id),
            alert_type,
            message,
            severity.lower(),
            is_resolved,
            resolved_at,
            now,
            now
        )

        with MySQLDatabase(**self.db_config) as db:
            last_id = db.execute_query(query, params, return_lastrowid=True)
            return last_id

    def get_alert(self, alert_id: int) -> Optional[Dict]:
        """
        Retrieve a single alert by ID
        
        Args:
            alert_id: ID of the alert to retrieve
            
        Returns:
            Dictionary with alert data or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE id = %s"
        
        with MySQLDatabase(**self.db_config) as db:
            results = db.execute_query(query, (alert_id,), fetch=True)
            return results[0] if results else None

    def get_alerts(
        self,
        device_id: Optional[Union[int, str]] = None,
        is_resolved: Optional[bool] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Retrieve multiple alerts with optional filters
        
        Args:
            device_id: Filter by device ID (optional)
            is_resolved: Filter by resolved status (optional)
            severity: Filter by severity level (optional)
            start_date: Filter alerts created after this date (optional)
            end_date: Filter alerts created before this date (optional)
            limit: Maximum number of alerts to return (default: 100)
            offset: Offset for pagination (default: 0)
            
        Returns:
            List of dictionaries with alert data
        """
        base_query = f"SELECT * FROM {self.table_name}"
        conditions = []
        params = []
        
        if device_id is not None:
            conditions.append("device_id = %s")
            params.append(str(device_id))
            
        if is_resolved is not None:
            conditions.append("is_resolved = %s")
            params.append(is_resolved)
            
        if severity is not None:
            conditions.append("severity = %s")
            params.append(severity.lower())
            
        if start_date is not None:
            conditions.append("created_at >= %s")
            params.append(start_date)
            
        if end_date is not None:
            conditions.append("created_at <= %s")
            params.append(end_date)
            
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
            
        base_query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(base_query, tuple(params), fetch=True) or []

    def update_alert(
        self,
        alert_id: int,
        **kwargs
    ) -> bool:
        """
        Update alert fields
        
        Args:
            alert_id: ID of the alert to update
            **kwargs: Fields to update (e.g., message='New message', is_resolved=True)
            
        Returns:
            True if update was successful, False otherwise
        """
        if not kwargs:
            return False
            
        set_clauses = []
        params = []
        
        for field, value in kwargs.items():
            if field.lower() == 'severity':
                value = value.lower()
            set_clauses.append(f"{field} = %s")
            params.append(value)
            
        params.append(alert_id)
        
        query = f"""
        UPDATE {self.table_name}
        SET {', '.join(set_clauses)}
        WHERE id = %s
        """
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params))

    def resolve_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as resolved with current timestamp
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if update was successful, False otherwise
        """
        return self.update_alert(
            alert_id,
            is_resolved=True,
            resolved_at=datetime.now()
        )

    def delete_alert(self, alert_id: int) -> bool:
        """
        Delete an alert record
        
        Args:
            alert_id: ID of the alert to delete
            
        Returns:
            True if successful, False otherwise
        """
        query = f"DELETE FROM {self.table_name} WHERE id = %s"
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, (alert_id,))

    def get_alert_stats(
        self,
        time_period: str = 'day',
        device_id: Optional[Union[int, str]] = None
    ) -> List[Dict]:
        """
        Get statistics about alerts
        
        Args:
            time_period: Grouping period ('hour', 'day', 'week', 'month')
            device_id: Filter by specific device (optional)
            
        Returns:
            List of alert statistics grouped by time period
        """
        valid_periods = ['hour', 'day', 'week', 'month']
        if time_period not in valid_periods:
            raise ValueError(f"Invalid time period. Must be one of: {valid_periods}")
            
        date_format = {
            'hour': '%Y-%m-%d %H:00:00',
            'day': '%Y-%m-%d',
            'week': '%Y-%u',
            'month': '%Y-%m'
        }[time_period]
        
        query = f"""
        SELECT 
            DATE_FORMAT(created_at, %s) as period,
            COUNT(*) as total_alerts,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_count,
            SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_count,
            SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_count,
            SUM(CASE WHEN severity = 'low' THEN 1 ELSE 0 END) as low_count,
            SUM(CASE WHEN is_resolved THEN 1 ELSE 0 END) as resolved_count
        FROM {self.table_name}
        """
        
        params = [date_format]
        
        if device_id is not None:
            query += " WHERE device_id = %s"
            params.append(str(device_id))
            
        query += " GROUP BY period ORDER BY period"
        
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params), fetch=True) or []
