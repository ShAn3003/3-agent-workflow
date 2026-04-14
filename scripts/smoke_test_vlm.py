"""图文结构输入场景的最小联调脚本。"""

import requests

payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"}},
            {"type": "text", "text": "请描述这张图，并标注不确定点。"}
        ]
    }]
}
s = requests.Session()
# 避免代理导致本地请求走外网。
s.trust_env = False
r = s.post("http://127.0.0.1:9000/solve", json=payload, timeout=300)
print(r.status_code)
print(r.text[:1000])
