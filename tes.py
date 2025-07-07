import requests
import json

def send_test_alert():
    """Send a test alert to the Laravel endpoint"""
    # API endpoint URL
    url = "https://pey.my.id/api/send-alert"
    
    # Request headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Dummy test data
    dummy_data = {
    "id": "F024F95AB9EC",
    "type": "humidity_over",
    "message": "Kelembapan lingkungan tinggi",
    "severity": "low"
}
    
    print("Sending test alert with data:")
    print(json.dumps(dummy_data, indent=2))
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(dummy_data))
        
        # Print the response details
        print("\nResponse:")
        print(f"Status Code: {response.status_code}")
        print("Headers:")
        print(response.headers)
        print("\nResponse Body:")
        try:
            print(json.dumps(response.json(), indent=2))
        except ValueError:
            print(response.text)
            
        return response
        
    except requests.exceptions.RequestException as e:
        print(f"\nError sending request: {e}")
        return None

if __name__ == "__main__":
    print("Starting test request...\n")
    send_test_alert()