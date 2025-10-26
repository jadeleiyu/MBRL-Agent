import requests
import json

url = "https://ms-w7gsz7wq-100041498772-sw.gw.ap-nanjing.ti.tencentcs.com/ms-w7gsz7wq/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 670604055d48b81"  
}
prompt='''
讲一个悲切而搞笑的故事
'''

payload = {
    "model": "ms-w7gsz7wq",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status:", response.status_code)
try:
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception:
    print("Raw text:")
    print(response.text)
