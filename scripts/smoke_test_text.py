import requests

payload = {
    "messages": [{"role": "user", "content": "解释 ReAct，并给一个 agent 场景例子。"}]
}
s = requests.Session()
s.trust_env = False
r = s.post("http://127.0.0.1:9000/solve", json=payload, timeout=300)
print(r.status_code)
print(r.text[:1000])
