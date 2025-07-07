import requests
import json

def send_alert_email(device_id, alert_type, message, severity):
    """
    Send an alert email by making a POST request to the Laravel endpoint.
    
    Parameters:
    - device_id: The ID of the device (e.g., "F024F95AB9EC")
    - alert_type: The type of alert (e.g., "humidity_over")
    - message: The alert message (e.g., "Kelembapan lingkungan tinggi")
    - severity: The severity level (e.g., "low")
    """
    # API endpoint URL
    url = "https://pey.my.id/api/send-alert"
    
    # Request headers (add any required headers like Authorization if needed)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Request payload
    payload = {
        "id": device_id,
        "type": alert_type,
        "message": message,
        "severity": severity
    }
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Alert sent successfully!")
            return response.json()
        else:
            print(f"Failed to send alert. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
