"""文本输入场景的最小联调脚本。"""

import requests

# 只传 `messages`，其余参数由 API 端走默认值。
payload = {
    "messages": [{"role": "user", "content": "解释 ReAct，并给一个 agent 场景例子。"}]
}
s = requests.Session()
# 避免系统代理干扰本地回环地址调用。
s.trust_env = False
r = s.post("http://127.0.0.1:9000/solve", json=payload, timeout=300)
print(r.status_code)
print(r.text[:1000])
