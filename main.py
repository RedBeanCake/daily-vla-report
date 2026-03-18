import requests
from bs4 import BeautifulSoup
import datetime
from openai import OpenAI
import os
import re
import json

# --- 核心配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")

# 自动解析仓库路径
repo_full_name = os.getenv('GITHUB_REPOSITORY', 'owner/repo')
repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'owner')
repo_name = repo_full_name.split('/')[-1]
GITHUB_PAGES_URL = f"https://{repo_owner}.github.io/{repo_name}/"

CATEGORIES = ['cs.RO']

def scrape_arxiv(category):
    """抓取 Arxiv 最新批次论文数据"""
    url = f"https://arxiv.org/list/{category}/recent?show=500"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        dls = soup.find_all('dl', id='articles')
        if not dls: return "Unknown", []
        
        target_date = soup.find('h3').text.strip()
        papers = []
        dt_tags = dls[0].find_all('dt')
        dd_tags = dls[0].find_all('dd')
        
        for dt, dd in zip(dt_tags, dd_tags):
            link_tag = dt.find('a', title='Abstract')
            if not link_tag: continue
            id_str = link_tag.text.replace('arXiv:', '').strip()
            title = dd.find('div', class_='list-title').text.replace('Title:', '').strip()
            abstract = dd.find('p', class_='mathjax').text.strip() if dd.find('p', class_='mathjax') else ""
            papers.append({
                "id": id_str, 
                "title": title, 
                "url": f"https://arxiv.org/pdf/{id_str}.pdf", 
                "abstract": abstract[:1000]
            })
        return target_date, papers
    except Exception: return None, []

def process_with_ai(papers, date_text):
    """调用大模型进行垂直领域筛选与提炼"""
    if not papers: return ""
    global_id = 1
    final_res = []
    
    for i in range(0, len(papers), 40):
        chunk = papers[i:i+40]
        prompt = f"""你是一个具身智能专家。请从以下论文中筛选并编号。
        ✅ 必须保留：
        1. VLA (Vision-Language-Action)、World Models (世界模型)、World Modeling、视频生成。
        2. World-Action Model、Video-Action Model、Diffusion Policy。
        3. 具身 Scaling Laws、跨具身数据集、多模态融合注意力。
        
        🛑 需要剔除：
        1. 经典控制(PID, MPC)、硬件/软体/步态研究。
        2. 传统导航、经典路径规划(A*, RRT)、SLAM、传感器标定。
        3. 垂直场景：深海、巡检、无人机/车、攀爬、医疗/手术机器人。
        4. 经典视觉：单纯的人体姿态识别、纯 3D 重建(NeRF/GS/三维几何)、单纯触觉。
        5. 多智能体协同/集群 (Swarm)、离散任务调度。
        
        要求：从编号 {global_id} 开始。
        格式要求：
        ### {global_id}. 🔥 [英文题目] (中文题目翻译)
        - **PDF链接**: [点击跳转]({{url}})
        - **核心亮点**: 一句话创新点。
        - **深度解析**: 一段话详细描述架构、数据、实验结论。
        """
        
        completion = client_llm.chat.completions.create(model="qwen-flash", messages=[{"role": "user", "content": prompt + str(chunk)}])
        res = completion.choices[0].message.content
        if "###" in res:
            final_res.append(res)
            global_id += res.count("###")
            
    return "\n\n---\n\n".join(final_res)

def generate_web_and_push(date_text, content):
    """构建 HTML 存档并推送飞书卡片"""
    count = content.count("###")
    
    # 构建 HTML 模板，采用 GitHub-Markdown 样式标准
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Embodied AI Report - {date_text}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            .markdown-body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }}
            @media (max-width: 767px) {{ .markdown-body {{ padding: 15px; }} }}
            .header-info {{ margin-bottom: 20px; color: #57606a; }}
        </style>
    </head>
    <body class="markdown-body">
        <h1>🤖 具身大模型简报 - {date_text}</h1>
        <div class="header-info">精选内容由 Qwen-flash 自动提炼 | 本日共有 <strong>{count}</strong> 篇核心论文</div>
        <hr>
        <div id="content"></div>
        <script>
            const rawMd = {json.dumps(content)};
            document.getElementById('content').innerHTML = marked.parse(rawMd);
        </script>
    </body>
    </html>
    """
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    
    # 构建飞书交互式卡片
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {{"title": {{"tag": "plain_text", "content": f"🌟 具身大模型精选 | {date_text}"}}, "template": "blue"}},
            "elements": [
                {{"tag": "div", "text": {{"tag": "lark_md", "content": f"今日已精选 **{count}** 篇 VLA 相关论文。详细提炼已更新至网页版存档。"}}}},
                {{"tag": "action", "actions": [
                    {{"tag": "button", "text": {{"tag": "plain_text", "content": "🌐 在线查阅网页版"}}, "type": "primary", "url": GITHUB_PAGES_URL}}
                ]}}
            ]
        }
    }
    requests.post(FEISHU_WEBHOOK, json=payload)

if __name__ == "__main__":
    all_p = {}
    actual_date = ""
    for cat in CATEGORIES:
        d, ps = scrape_arxiv(cat)
        if d: actual_date = d
        for p in ps: all_p[p['id']] = p
    
    final_list = list(all_p.values())
    content = process_with_ai(final_list, actual_date)
    if content: generate_web_and_push(actual_date, content)
