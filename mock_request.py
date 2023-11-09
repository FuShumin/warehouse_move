# your_script.py

import requests
import json

# Step 2: Set up the URL and payload
url = "http://localhost:5000/get_final_report"
payload = {
    "params": {
        "client_id": "1691005891402997762"
    }
}

# Step 3: Perform the HTTP POST request
response = requests.post(url, json=payload)

# Step 4: Print the server's response
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=4))
