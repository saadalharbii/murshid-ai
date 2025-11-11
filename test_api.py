"""Quick test script for the MurshidAI API"""
import requests
import json

API_URL = "http://127.0.0.1:8000/api/query"

# Test English query
print("=" * 60)
print("Testing English Query")
print("=" * 60)
response = requests.post(API_URL, json={"question": "What are the best scholarship programs?"})
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Test Arabic query
print("\n" + "=" * 60)
print("Testing Arabic Query")
print("=" * 60)
response = requests.post(API_URL, json={"question": "ما هي أفضل الجامعات؟"})
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# Test health endpoint
print("\n" + "=" * 60)
print("Testing Health Endpoint")
print("=" * 60)
response = requests.get("http://127.0.0.1:8000/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
