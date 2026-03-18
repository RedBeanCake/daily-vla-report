import requests
from bs4 import BeautifulSoup
import datetime
from openai import OpenAI
import os
import re
import json

# --- 1. 配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 从系统环境变量读取
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
GITHUB_PAGES_URL = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER')}.github.io/{os.getenv('GITHUB_REPOSITORY')}/"

# CATEGORIES = ['cs.RO', 'cs.AI', 'cs.CV', 'cs.LG']
CATEGORIES = ['cs.RO']

def scrape_arxiv(category):
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
            id_str = dt.find('a', title='Abstract').text.replace('arXiv:', '').strip()
            title = dd.find('div', class_='list-title').text.replace('Title:', '').strip()
            abstract = dd.find('p', class_='mathjax').text.strip() if dd.find('p', class_='mathjax') else ""
            papers.append({"id": id_str, "title": title, "url": f"https://arxiv.org/pdf/{id_str}.pdf", "abstract": abstract[:1000]})
        return target_date, papers
    except: return None, []

def process_with_ai(papers, date_text):
    if not papers: return ""
    global_id = 1
    final_res = []
    
    # 分块处理 (40篇一组)
    for i in range(0, len(papers), 40):
        chunk = papers[i:i+40]
        prompt = f"""你是一个具身智能专家。请从以下论文中筛选并编号。
        ✅ 必须保留：Vision-Language-Action (VLA)、World Models、World Modeling、视频生成、Video-Action Model、World-Action Model、Diffusion Policy、Action Chunking (ACT)、具身数据 Scaling Laws、跨具身数据集。
        🛑 需要剔除：经典控制(PID, MPC)、硬件/软体/步态研究、传统导航/SLAM/路径规划、无人机/车、深海、巡检、攀爬、医疗/手术机器人、人体姿态识别、纯触觉、多智能体协同/集群 (Swarm)、离散任务调度。
        
        要求：从编号 {global_id} 开始。
        格式：### {global_id}. 🔥 [英文题目] (中文题目翻译)
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
    count = content.count("###")
    # 生成 HTML
    html_body = content.replace('###', '<h3>').replace('\n- ', '<li>').replace('🔥', '<span style="color:#e67e22">🔥</span>')
    html_template = f"<html><head><meta charset='utf-8'><link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/water.css@2/out/light.css'><title>VLA Report</title></head><body><h1>🤖 具身大模型简报 - {date_text}</h1><p>精选: {count} 篇</p>{html_body}</body></html>"
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    
    # 推送飞书卡片
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"🌟 具身大模型精选 | {date_text}"}, "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"今日精选 **{count}** 篇 VLA 相关论文。已自动更新至网页版。"}},
                {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "🌐 在线查阅网页版"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
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
