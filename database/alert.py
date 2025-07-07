from datetime import datetime
from typing import List, Dict, Optional, Union
from .db import MySQLDatabase
import logging

class AlertManager:
    def __init__(self, db_config: dict, table_name: str = 'alerts'):
        self.db_config = db_config
        self.table_name = table_name
        self.logger = logging.getLogger(__name__)
        self._verify_table_structure()


    def _verify_table_structure(self):
        print("8.1")
        with MySQLDatabase(**self.db_config) as db:
            print("8.2")
            if not db.table_exists(self.table_name):
                print("8.3")
                self.logger.warning(f"Table '{self.table_name}' does not exist")
                # Optional: db.execute_query(self._create_table_sql())
            print("8.4")

    def _create_table_sql(self) -> str:
        return f"""
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
    def create_alert(
    self,
    device_id: Union[str, int],
    type: str,  # This is the parameter name
    message: str,
    severity: str,
    is_resolved: bool = False,
    resolved_at: Optional[datetime] = None
) -> Optional[int]:
        today = datetime.now().date()
        start_of_day = datetime.combine(today, datetime.min.time())
        end_of_day = datetime.combine(today, datetime.max.time())
        
        existing_alerts = self.get_alerts(
            device_id=device_id,
            alert_type=type,  # Changed from 'type' to 'alert_type' to match get_alerts parameter
            is_resolved=False,
            start_date=start_of_day,
            end_date=end_of_day     
        )
        print(existing_alerts)
        if len(existing_alerts)>0:
            self.logger.info(f"Duplicate alert found for device {device_id} (type: {type}) - skipping creation")
            # return None
        else:
            # If no existing alert, proceed with creation
            query = f"""
            INSERT INTO {self.table_name} 
            (device_id, type, message, severity, is_resolved, resolved_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            now = datetime.now()
            params = (
                str(device_id),
                type,
                message,
                severity.lower(),
                is_resolved,
                resolved_at,
                now,
                now
            )

            with MySQLDatabase(**self.db_config) as db:
                db.execute_query(query, params)

    def get_alerts(
        self,
        device_id: Optional[Union[str, int]] = None,
        alert_type: Optional[str] = None,  # This is the parameter name we need to match
        is_resolved: Optional[bool] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        query = f"SELECT * FROM {self.table_name}"
        conditions = []
        params = []

        if device_id:
            conditions.append("device_id = %s")
            params.append(str(device_id))
        if alert_type:  # Using 'alert_type' consistently
            conditions.append("type = %s")  # Note: 'type' is the column name in SQL
            params.append(alert_type)
        if is_resolved is not None:
            conditions.append("is_resolved = %s")
            params.append(is_resolved)
        if severity:
            conditions.append("severity = %s")
            params.append(severity.lower())
        if start_date:
            conditions.append("created_at >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= %s")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params), fetch=True) or []
        
    def get_alert(self, alert_id: int) -> Optional[Dict]:
        query = f"SELECT * FROM {self.table_name} WHERE id = %s"
        with MySQLDatabase(**self.db_config) as db:
            result = db.execute_query(query, (alert_id,), fetch=True)
            return result[0] if result else None

    def update_alert(self, alert_id: int, **kwargs) -> bool:
        if not kwargs:
            return False

        set_clauses = []
        params = []

        for key, value in kwargs.items():
            if key == 'severity':
                value = value.lower()
            set_clauses.append(f"{key} = %s")
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
        return self.update_alert(
            alert_id,
            is_resolved=True,
            resolved_at=datetime.now()
        )

    def delete_alert(self, alert_id: int) -> bool:
        query = f"DELETE FROM {self.table_name} WHERE id = %s"
        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, (alert_id,))

    def get_alert_stats(
        self,
        time_period: str = 'day',
        device_id: Optional[Union[str, int]] = None
    ) -> List[Dict]:
        period_format = {
            'hour': '%Y-%m-%d %H:00:00',
            'day': '%Y-%m-%d',
            'week': '%Y-%u',
            'month': '%Y-%m'
        }.get(time_period)

        if not period_format:
            raise ValueError("Invalid time period. Use 'hour', 'day', 'week', or 'month'")

        query = f"""
        SELECT 
            DATE_FORMAT(created_at, %s) AS period,
            COUNT(*) AS total_alerts,
            SUM(severity = 'critical') AS critical_count,
            SUM(severity = 'high') AS high_count,
            SUM(severity = 'medium') AS medium_count,
            SUM(severity = 'low') AS low_count,
            SUM(is_resolved) AS resolved_count
        FROM {self.table_name}
        """
        params = [period_format]

        if device_id:
            query += " WHERE device_id = %s"
            params.append(str(device_id))

        query += " GROUP BY period ORDER BY period"

        with MySQLDatabase(**self.db_config) as db:
            return db.execute_query(query, tuple(params), fetch=True) or []
