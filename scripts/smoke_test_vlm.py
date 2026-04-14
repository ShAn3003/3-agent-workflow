"""图文结构输入场景的最小联调脚本。"""

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
IMAGES = [
    ROOT / "assets" / "test_images" / "realworld_04.png",
    ROOT / "assets" / "test_images" / "realworld_03.png",
]

s = requests.Session()
# 避免代理导致本地请求走外网。
s.trust_env = False
for img in IMAGES:
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{img}"}},
                {"type": "text", "text": "请描述这张图，并标注不确定点。"}
            ]
        }]
    }
    r = s.post("http://127.0.0.1:9000/solve", json=payload, timeout=300)
    print(f"[image] {img.name}")
    print(r.status_code)
    print(r.text[:1000])
