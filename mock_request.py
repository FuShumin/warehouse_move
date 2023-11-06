import requests
import json

try:
    # url = "http://10.1.21.201:35000/get_final_report"
    # url = "http://10.1.20.196:5000/get_final_report"
    # url = "http://127.0.0.1:5000/get_final_report"
    url = "http://10.1.20.120:5000/ocr"
    # url = "http://127.0.0.1:5000/ocr"
    # payload = {
    #     "params": {
    #         "pageId": "undefined",
    #         "regulation_name": "A",
    #         "deleted": 0,
    #         "rows": 20,
    #         "page": 1,
    #         "appCode": "APP202310201337131",
    #         "app_code": "APP202310201337131",
    #         "clientId": "1691005891402997762",
    #         "client_id": "1691005891402997762"
    #     },
    #     "orderByItem": [
    #         {
    #             "field": "send_name",
    #             "order": "desc"
    #         }
    #     ],
    #     "pageQuery": {
    #         "rows": 20,
    #         "page": 1,
    #         "order": "",
    #         "sort": ""
    #     }
    # }
    payload = {
            "img_path": "/usr/src/ultralytics/detect/mpeg4.JPG"
    }

    response = requests.post(url, json=payload)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Unexpected status code: {response.status_code}. Response content: {response.content}")

except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON response.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
