import paho.mqtt.client as mqtt
import json
from .message_handler import MessageHandler
import logging
from threading import Lock
from datetime import datetime
class MQTTHandler:
    def __init__(self, broker='broker.hivemq.com', port=1883, monitoring_topic='iot/monitoring', device_control_topic='smartpower/device/control', device_status_topic='smartpower/device/status', 
    device_alert_topic='smartpower/device/alert',
    control_topic_prefix=None):
        DB_CONFIG = {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',
            'database': 'peymyid_pey'
        }
        # DB_CONFIG = {
        #     'host': 'localhost',
        #     'user': 'peymyid_pey',
        #     'password': 'Pey12345.#@',
        #     'database': 'peymyid_pey'
        # }
        self.handler = MessageHandler(DB_CONFIG)
        self.broker = broker
        self.port = port
        self.monitoring_topic = monitoring_topic
        self.device_control_topic = device_control_topic
        self.device_status_topic = device_status_topic
        self.device_alert_topic = device_alert_topic
        self.control_topic_prefix = control_topic_prefix
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
                (f"{self.device_control_topic}/#", 1),
                (f"{self.device_status_topic}/#", 1),
                (f"{self.device_alert_topic}/#", 1)
            ]
            client.subscribe(topics)
            self.logger.info(f"Subscribed to topics: {[t[0] for t in topics]}")
        else:
            self.logger.error(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            self.logger.info(f"Received message from {msg.topic}: {payload}")
            self.handler.handle_message(msg.topic, payload)
            try:
                # Try to parse as JSON
                data = json.loads(payload)
            except json.JSONDecodeError:
                # If not JSON, store as raw string
                data = payload
                
            with self.data_lock:
                self.latest_data[msg.topic] = {
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

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

    def publish(self, topic, message, retain=False, qos=1):
        """Publish message to MQTT topic"""
        try:
            if not self.connected:
                self.logger.warning ("Not connected to MQTT broker. Attempting to reconnect...")
                self.connect()
                
            if not isinstance(message, str):
                message = json.dumps(message)
                
            result = self.client.publish(topic, message, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"Published to {topic}: {message}")
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
                # Get specific topic data
                return self.latest_data.get(topic)
            else:
                # Get all data
                return self.latest_data.copy()