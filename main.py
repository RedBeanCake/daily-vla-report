import requests
from bs4 import BeautifulSoup
import datetime
from openai import OpenAI
import os
import re
import json

# --- 1. 核心配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")

# 动态解析 GitHub 仓库信息
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
        if not dls: return None, []
        
        target_date = soup.find('h3')[1].text.strip()
        papers = []
        dt_tags = dls[1].find_all('dt')
        dd_tags = dls[1].find_all('dd')
        
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
    """调用大模型进行结构化筛选与提炼"""
    if not papers: return ""
    global_id_counter = 1
    final_res = []
    
    # 分块处理 (40篇一组)
    for i in range(0, len(papers), 40):
        chunk = papers[i:i+40]
        prompt = f"""你是一个专注于【大模型具身智能】的顶级研究员。请从以下论文中筛选并编号。

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

        要求：
        1. 请从编号 {global_id_counter} 开始为筛选出的论文连续编号。
        2. 格式（Markdown）：
           ### {global_id_counter}. 🔥 [英文题目] (中文题目翻译)
           - **PDF链接**: [点击跳转]({{url}})
           - **核心亮点**: (一句话说明该文最核心的创新点)。
           - **深度解析**: (一段话详细描述技术方案、训练数据规模、实验结论及物理意义)。
           - **领域归类**: [归类版块]

        待处理数据内容：
        {json.dumps(chunk)}
        """
        
        try:
            completion = client_llm.chat.completions.create(
                model="qwen-flash", 
                messages=[{"role": "user", "content": prompt}]
            )
            res = completion.choices[0].message.content
            if "###" in res:
                final_res.append(res)
                global_id_counter += res.count("###")
        except Exception as e:
            print(f"AI Error: {e}")
            
    return "\n\n---\n\n".join(final_res)

def generate_archive_and_index(date_text, content):
    """生成每日 HTML 详情页并动态更新主索引页 (优化 Safari 渲染)"""
    count = content.count("###")
    safe_date = re.sub(r'[^\w\s-]', '', date_text).replace(' ', '_')
    
    os.makedirs('archive', exist_ok=True)
    daily_file_path = f"archive/{safe_date}.html"

    def get_html_template(title, body_content, is_index=False):
        back_link = "<a href='../index.html' style='margin-bottom:20px; display:block;'>← 返回主索引</a>" if not is_index else ""
        return f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>{title}</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <style>
                /* 针对 Safari 的字体优化 */
                :root {{
                    color-scheme: light; /* 强制明亮模式，防止 Safari 自动反色导致样式错乱 */
                }}
                .markdown-body {{
                    box-sizing: border-box;
                    min-width: 200px;
                    max-width: 980px;
                    margin: 0 auto;
                    padding: 45px;
                    /* 修复 Safari 字体显示的核心代码 */
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
                    -webkit-font-smoothing: antialiased; /* 开启平滑锯齿 */
                    text-rendering: optimizeLegibility;
                }}
                @media (max-width: 767px) {{
                    .markdown-body {{ padding: 15px; }}
                }}
                /* 针对 Safari 的链接颜色微调 */
                .markdown-body a {{ color: #0969da; text-decoration: none; }}
                .markdown-body a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body class="markdown-body">
            {back_link}
            <h1>{title}</h1>
            <div id="content"></div>
            <script>
                const rawMd = {json.dumps(body_content)};
                document.getElementById('content').innerHTML = marked.parse(rawMd);
            </script>
        </body>
        </html>
        """

    # 1. 保存每日详情页
    with open(daily_file_path, "w", encoding="utf-8") as f:
        f.write(get_html_template(f"🤖 具身大模型简报 - {date_text}", content))

    # 2. 扫描 archive 目录并生成 index.html
    history_files = sorted([f for f in os.listdir('archive') if f.endswith('.html')], reverse=True)
    index_md = "### 📅 历史存档列表\n\n"
    for f_name in history_files:
        display_date = f_name.replace('.html', '').replace('_', ' ')
        index_md += f"- [{display_date}](archive/{f_name})\n"
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(get_html_template("📚 具身大模型科研日报 - 历史索引", index_md, is_index=True))

    # 3. 推送飞书卡片
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"🌟 具身大模型精选 | {date_text}"}, "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"今日已精选 **{count}** 篇 VLA 相关论文。详情已归档至历史索引页。"}},
                {"tag": "action", "actions": [
                    {"tag": "button", "text": {"tag": "plain_text", "content": "🌐 进入历史索引查阅"}, "type": "primary", "url": GITHUB_PAGES_URL}
                ]}
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
    if content: generate_archive_and_index(actual_date, content)
