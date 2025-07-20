import json
import logging
from datetime import datetime
from threading import Lock

import paho.mqtt.client as mqtt

from utils.device_monitor import DeviceMonitor

from .message_handler import MessageHandler


class MQTTHandler:
    def __init__(self, broker='broker.hivemq.com', port=1883, monitoring_topic='iot/monitoring', 
                 device_control_topic='smartpower/device/control', device_status_topic='smartpower/device/status',device_sensor_topic='smartpower/device/sensor',
                 device_alert_topic='smartpower/device/alert', control_topic_prefix=None):
        # Initialize database configuration
        with open('config.json', 'r') as f:
            DB_CONFIG = json.load(f)

        
        # Initialize message handler first
        self.handler = MessageHandler(DB_CONFIG)
        
        # Then initialize device monitor with the device manager
        self.monitor = DeviceMonitor(self.handler.device_manager)
        
        # MQTT configuration
        self.broker = broker
        self.port = port
        self.monitoring_topic = monitoring_topic
        self.device_control_topic = device_control_topic
        self.device_status_topic = device_status_topic
        self.device_sensor_topic = device_sensor_topic
        self.device_alert_topic = device_alert_topic
        self.control_topic_prefix = control_topic_prefix
        
        # MQTT client setup
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.latest_data = {}
        self.data_lock = Lock()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.logger.info("Connected to MQTT Broker!")
            # Subscribe to multiple topics
            topics = [
                (f"{self.monitoring_topic}/#", 1),
                # (f"{self.device_control_topic}/#", 1),
                (f"{self.device_status_topic}/#", 1),
                (f"{self.device_sensor_topic}/#", 1),
                (f"{self.device_alert_topic}/#", 1)
            ]
            client.subscribe(topics)
            self.logger.info(f"Subscribed to topics: {[t[0] for t in topics]}")
            
            # Reset monitoring on new connection
            self.reset_device_monitoring()
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            self.logger.debug(f"Received message from {msg.topic}: {payload[:200]}...")
            
            # Handle message through the message handler
            self.handler.handle_message(msg.topic, payload)
            
            try:
                # Try to parse as JSON
                data = json.loads(payload)
                
                # Register active device if ID exists in the message
                if 'id' in data:
                    device_id = data['id']
                    self.monitor.add_active_device(device_id)
                    self.logger.debug(f"Registered active device: {device_id}")
                    
            except json.JSONDecodeError:
                # If not JSON, store as raw string
                data = payload
                
            # Store message data
            with self.data_lock:
                self.latest_data[msg.topic] = {
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)

    def reset_device_monitoring(self):
        """
        Reset the device monitoring system
        1. Mark all existing devices as inactive
        2. Clear the monitoring lists
        3. Prepare for fresh monitoring cycle
        """
        self.logger.info("Resetting device monitoring system...")
        
        try:
            # Safely get all device IDs (handle case where method doesn't exist)
            if hasattr(self.handler.device_manager, 'get_all_device_ids'):
                all_devices = self.handler.device_manager.get_all_device_ids()
            else:
                self.logger.warning("DeviceManager missing get_all_device_ids method")
                all_devices = []
            
            if all_devices:
                # Mark all devices as inactive
                if hasattr(self.handler.device_manager, 'bulk_update_state'):
                    updated = self.handler.device_manager.bulk_update_state(all_devices, 'inactive')
                    self.logger.info(f"Marked {updated} devices as inactive in database")
                else:
                    self.logger.warning("DeviceManager missing bulk_update_state method")
            
            # Reset the monitor
            if hasattr(self.monitor, 'reset_all_devices'):
                self.monitor.reset_all_devices()
                self.logger.info("Device monitoring has been reset")
            else:
                self.logger.warning("DeviceMonitor missing reset_all_devices method")
                
        except Exception as e:
            self.logger.error(f"Error during monitoring reset: {e}", exc_info=True)

    def force_check_inactive_devices(self):
        """
        Manually trigger inactive device check
        Useful for testing or maintenance
        """
        self.logger.info("Manually triggering inactive device check")
        self.monitor._check_inactive_devices()

    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.logger.info(f"Disconnected from MQTT Broker (rc: {rc})")
        if rc != 0:
            self.logger.warning("Unexpected disconnection. Trying to reconnect...")
            self.connect()

    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            self.logger.error(f"MQTT connection error: {e}")

    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        self.logger.info("MQTT client disconnected")

    def publish(self, topic, message, retain=False, qos=1):
        """Publish message to MQTT topic"""
        try:
            if not self.connected:
                self.logger.warning("Not connected to MQTT broker. Attempting to reconnect...")
                self.connect()
                
            if not isinstance(message, str):
                message = json.dumps(message)
                
            result = self.client.publish(topic, message, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"Published to {topic}: {message[:200]}...")
                return True
            else:
                self.logger.error(f"Failed to publish to {topic}. Error code: {result.rc}")
                return False
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            return False

    def get_latest_data(self, topic=None):
        """Get latest MQTT data"""
        with self.data_lock:
            if topic:
                return self.latest_data.get(topic)
            return self.latest_data.copy()

    def get_active_devices(self):
        """Get list of currently active devices"""
        return self.monitor.current_devices

    def get_inactive_devices(self):
        """Get list of recently inactive devices"""
        return self.monitor.previous_devices - self.monitor.current_devices