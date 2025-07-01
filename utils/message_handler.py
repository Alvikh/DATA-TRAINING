import paho.mqtt.client as mqtt
import json
import logging
from threading import Lock
from datetime import datetime
from database.device import DeviceManager
from database.alert import AlertManager
class MessageHandler:
    def __init__(self, db_config: dict):
        """
        Initialize message handler with database configuration
        
        Args:
            db_config: Database configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.MessageHandler")
        self.device_manager = DeviceManager(db_config)
        self.alert_manager = AlertManager(db_config)
        self.latest_data = {}
        self.data_lock = Lock()

    def handle_message(self, topic: str, payload: str) -> bool:
        """
        Main message handler that processes incoming MQTT messages
        
        Args:
            topic: MQTT topic the message was received on
            payload: Raw message payload
            
        Returns:
            bool: True if message was handled successfully
        """
        try:
            self.logger.debug(f"Processing message on {topic}: {payload[:100]}...")
            
            # Parse payload
            try:
                payload_data = json.loads(payload)
            except json.JSONDecodeError:
                payload_data = payload
            
            # Route to appropriate handler
            if "smartpower/device/status" in topic:
                return self._handle_status_message(topic, payload_data)
            # elif "smartpower/device/control" in topic:
            #     return self._handle_control_message(topic, payload_data)
            elif "smartpower/device/alert" in topic:
                return self._handle_alert_message(topic, payload_data)
            elif "iot/monitoring" in topic:
                return self._handle_monitoring_message(topic, payload_data)
            else:
                self.logger.warning(f"No handler for topic: {topic}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}", exc_info=True)
            return False

    def _handle_status_message(self, topic: str, payload: dict) -> bool:
        """Handle device status updates"""
        try:
            required_fields = ['id', 'state']
            if not all(field in payload for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")
            
            device_id = payload['id']
            new_state = str(payload['state']).strip()
            db_state = new_state

            success = self.device_manager.update_device_state(device_id, db_state)
            
            if success:
                with self.data_lock:
                    self.latest_data[topic] = {
                        'device_id': device_id,
                        'state': db_state,
                        'status': payload.get('status', 'unknown'),
                        'timestamp': datetime.now().isoformat(),
                        'original_payload': payload
                    }
                self.logger.info(f"Updated device {device_id} state to {db_state}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"message handler class error: {e}")
            return False

    def _handle_control_message(self, topic: str, payload: dict) -> bool:
        """Handle control commands"""
        try:
            with self.data_lock:
                self.latest_data[topic] = {
                    'command': payload.get('command'),
                    'target': payload.get('target'),
                    'timestamp': datetime.now().isoformat(),
                    'original_payload': payload
                }
            self.logger.info(f"Processed control command for {payload.get('target')}")
            return True
        except Exception as e:
            self.logger.error(f"Control message error: {e}")
            return False
    def _handle_alert_message(self, topic: str, payload: dict) -> bool:
        """Handle alert messages and store them in the database"""
        try:
            # Validate required fields
            required_fields = ['device_id', 'type', 'message', 'severity']
            if not all(field in payload for field in required_fields):
                raise ValueError(f"Missing required fields in alert: {required_fields}")

            # Extract alert data
            device_id = payload['device_id']
            alert_type = payload['type']
            message = payload['message']
            severity = payload['severity'].lower()
            
            # Validate severity level
            valid_severities = ['low', 'medium', 'high', 'critical']
            if severity not in valid_severities:
                severity = 'medium'  # Default to medium if invalid
                self.logger.warning(f"Invalid severity level, defaulting to 'medium'. Received: {severity}")

            # Create alert in database
            alert_id = self.alert_manager.create_alert(
                device_id=device_id,
                alert_type=alert_type,
                message=message,
                severity=severity,
                is_resolved=False
            )

            if alert_id:
                with self.data_lock:
                    self.latest_data[topic] = {
                        'alert_id': alert_id,
                        'device_id': device_id,
                        'type': alert_type,
                        'severity': severity,
                        'timestamp': datetime.now().isoformat(),
                        'original_payload': payload
                    }
                self.logger.info(f"Created new alert (ID: {alert_id}) for device {device_id}")
                return True
            
            self.logger.error(f"Failed to create alert in database: {alert_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing alert message: {e}", exc_info=True)
            return False
    def _handle_monitoring_message(self, topic: str, payload: dict) -> bool:
        """Handle telemetry data"""
        try:
            with self.data_lock:
                self.latest_data[topic] = {
                    'data': payload,
                    'timestamp': datetime.now().isoformat()
                }
            self.logger.debug(f"Stored telemetry data from {topic}")
            return True
        except Exception as e:
            self.logger.error(f"Monitoring message error: {e}")
            return False

    def get_latest_data(self, topic: str = None) -> dict:
        """Get latest processed message data"""
        with self.data_lock:
            if topic:
                return self.latest_data.get(topic)
            return self.latest_data.copy()