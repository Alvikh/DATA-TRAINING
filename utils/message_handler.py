import paho.mqtt.client as mqtt
import json
import logging
from threading import Lock
from datetime import datetime
from database.device import DeviceManager
from database.alert import AlertManager
from database.energy_measurement import EnergyMeasurement  # tambahkan ini di atas

class MessageHandler:
    def __init__(self, db_config: dict):
        print(">>> 6")
        self.logger = logging.getLogger(f"{__name__}.MessageHandler")
        print(">>> 7")
        self.device_manager = DeviceManager(db_config)
        print(">>> 8")
        self.alert_manager = AlertManager(db_config)
        print(">>> 9")
        self.energy_measurement = EnergyMeasurement(db_config)  # 🔧 TAMBAHKAN INI
        print(">>> 10")
        self.latest_data = {}
        print(">>> 11")
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
            elif "smartpower/device/control" in topic:
                return self._handle_control_message(topic, payload_data)
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
                type=alert_type,
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
                
                # Send alert email notification
                try:
                    from api.alert import send_alert_email
                    email_response = send_alert_email(
                        device_id=device_id,
                        alert_type=alert_type,
                        message=message,
                        severity=severity
                    )
                    
                    if email_response:
                        self.logger.info(f"Alert email sent successfully. Response: {email_response}")
                    else:
                        self.logger.warning("Failed to send alert email")
                except ImportError:
                    self.logger.error("Could not import send_alert_email from api.alert")
                except Exception as e:
                    self.logger.error(f"Error sending alert email: {e}")
                
                return True
            
            self.logger.error(f"Failed to create alert in database: {alert_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing alert message: {e}", exc_info=True)
            return False
        
    def _handle_monitoring_message(self, topic: str, payload: dict) -> bool:
        """Handle telemetry data and store in database"""
        try:
            # 1. Extract and validate device ID
            device_id = payload.get('id', '').strip()
            if not device_id:
                self.logger.error("No device ID in payload")
                return False

            # 2. Prepare and sanitize measurement data
            measured_at = payload.get('measured_at')
            try:
                if measured_at:
                    # Convert string timestamp to datetime object then format for MySQL
                    measured_at = datetime.strptime(measured_at, '%Y-%m-%d %H:%M:%S')
                else:
                    measured_at = datetime.now()
            except Exception as e:
                self.logger.warning(f"Invalid timestamp format, using current time: {e}")
                measured_at = datetime.now()

            measurement_data = {
                'device_id': device_id,
                'voltage': float(payload.get('voltage', 0)) if payload.get('voltage') is not None else None,
                'current': float(payload.get('current', 0)) if payload.get('current') is not None else None,
                'power': float(payload.get('power', 0)) if payload.get('power') is not None else None,
                'energy': float(payload.get('energy', 0)) if payload.get('energy') is not None else None,
                'frequency': float(payload.get('frequency', 0)) if payload.get('frequency') is not None else None,
                'power_factor': float(payload.get('power_factor', 0)) if payload.get('power_factor') is not None else None,
                'temperature': float(payload.get('temperature')) if payload.get('temperature') is not None else None,
                'humidity': float(payload.get('humidity')) if payload.get('humidity') is not None else None,
                'measured_at': measured_at.strftime('%Y-%m-%d %H:%M:%S')  # MySQL format
            }

            self.logger.debug(f"Prepared measurement data: {measurement_data}")

            # 3. Verify device exists before insertion
            if not self.device_manager.device_exists(device_id):
                self.logger.error(f"Device {device_id} does not exist in database")
                return False

            # 4. Store to database
            measurement_id = self.energy_measurement.create(measurement_data)
            
            measurement_id = self.energy_measurement.create(measurement_data)
            if not measurement_id:
                self.logger.error(f"Insert failed. Data: {measurement_data}")
                return False
            else:
                self.logger.info(f"Insert success. ID: {measurement_id}")

                
            # 5. Update in-memory cache
            with self.data_lock:
                self.latest_data[topic] = {
                    'data': payload,
                    'timestamp': datetime.now().isoformat(),
                    'db_id': measurement_id
                }
                
            self.logger.info(f"Successfully stored telemetry from {device_id} (ID: {measurement_id})")
            return True
            
        except ValueError as ve:
            self.logger.error(f"Data conversion error: {ve}", exc_info=True)
            return False
        except KeyError as ke:
            self.logger.error(f"Missing key in payload: {ke}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error processing message: {e}", exc_info=True)
            return False

    def get_latest_data(self, topic: str = None) -> dict:
        """Get latest processed message data"""
        with self.data_lock:
            if topic:
                return self.latest_data.get(topic)
            return self.latest_data.copy()
        