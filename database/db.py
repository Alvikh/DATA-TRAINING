import mysql.connector
from mysql.connector import Error
import logging
from typing import Optional, Union, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MySQLDatabase:
    """
    A comprehensive MySQL database connection handler with CRUD operations
    
    Attributes:
        config (dict): Database connection parameters
        connection: MySQL connection object
    """
    
    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        """
        Initialize MySQL database connection
        
        Args:
            host: Database host address
            user: Database username
            password: Database password
            database: Database name
            port: Database port (default: 3306)
        """
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'port': port,
            'autocommit': False,  # Better to control transactions manually
            'charset': 'utf8mb4',  # Supports full Unicode including emojis
            'collation': 'utf8mb4_unicode_ci',
        }
        self.connection = None

    def __enter__(self):
        """Context manager entry - establishes connection"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed"""
        self.disconnect()

    def connect(self) -> bool:
        """
        Establish a connection to the MySQL database
        
        Returns:
            bool: True if connection succeeded, False otherwise
        """
        try:
            self.connection = mysql.connector.connect(**self.config)
            
            # Test the connection
            if self.connection.is_connected():
                db_info = self.connection.get_server_info()
                logger.info(f"Connected to MySQL Server version {db_info}")
                return True
            
        except Error as e:
            logger.error(f"Error connecting to MySQL database: {e}")
            self.connection = None
            return False

    def disconnect(self):
        """Close the database connection if it exists"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
            self.connection = None

    def execute_query(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        fetch: bool = False,
        commit: bool = True,
        return_lastrowid: bool = False
    ) -> Union[bool, List[Dict], int, None]:
        cursor = None
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            logger.debug(f"Executing query: {query}")
            logger.debug(f"With parameters: {params}")
            
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
                logger.debug(f"Fetched {len(result)} rows")
                return result
            
            if commit:
                self.connection.commit()
                logger.debug("Transaction committed")
            
            if return_lastrowid:
                return cursor.lastrowid
            
            return True
            
        except Error as e:
            logger.error(f"Error executing query: {e}")
            if self.connection:
                self.connection.rollback()
                logger.info("Transaction rolled back")
            return None if return_lastrowid else False
            
        finally:
            if cursor:
                cursor.close()


    def get_tables(self) -> List[str]:
        """Get list of tables in the database"""
        query = "SHOW TABLES"
        result = self.execute_query(query, fetch=True)
        return [list(table.values())[0] for table in result] if result else []

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        query = """
        SELECT COUNT(*) as count 
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
        """
        result = self.execute_query(query, (self.config['database'], table_name), fetch=True)
        return result[0]['count'] > 0 if result else False

    def get_table_columns(self, table_name: str) -> List[Dict]:
        """Get column information for a table"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default, column_key
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """
        return self.execute_query(query, (self.config['database'], table_name), fetch=True) or []
