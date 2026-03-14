import requests

url = "http://127.0.0.1:8000/predict"
image_path = "/Users/jois-mba/Developer/xray.jpg"   # change this to your image

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)
