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

repo_full_name = os.getenv('GITHUB_REPOSITORY', 'owner/repo')
repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'owner')
repo_name = repo_full_name.split('/')[-1]
GITHUB_PAGES_URL = f"https://{repo_owner}.github.io/{repo_name}/"

CATEGORIES = ['cs.RO']

def scrape_hf_daily():
    """抓取 Hugging Face Daily Papers 原始数据"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        res = requests.get("https://huggingface.co/api/daily_papers", headers=headers, timeout=15)
        if res.status_code != 200: return []
        return res.json()
    except Exception as e:
        print(f"HF Scrape Error: {e}")
        return []

def process_hf_with_ai(hf_papers):
    """使用 AI 为 HF 论文生成亮点、解析和归类，并按点赞数排序"""
    if not hf_papers or not isinstance(hf_papers, list): return ""
    
    # 1. 提取关键信息给 AI，并进行安全加固
    simple_list = []
    for p in hf_papers:
        paper_info = p.get('paper', {})
        if not paper_info or 'id' not in paper_info:
            continue
            
        # 兼容性获取点赞数：尝试从外层或内层获取
        upvotes_val = p.get('upvotes') or paper_info.get('upvotes', 0)
        
        simple_list.append({
            "id": paper_info.get('id', ''),
            "title": paper_info.get('title', 'Unknown Title'),
            "upvotes": upvotes_val
        })

    # 2. 【核心修改】：按照点赞数（upvotes）从高到低排序
    simple_list.sort(key=lambda x: x.get('upvotes', 0), reverse=True)

    # 3. 构造 Prompt，明确要求 AI 保持排序顺序
    prompt = f"""你是一个 AI 大模型专家。请为以下 Hugging Face 每日热门论文提供中文解析。
    
    要求：
    1. **不要剔除**任何论文，全部保留。
    2. **严格保持列表给出的顺序**（已经按点赞数从高到低排列），不要打乱。
    3. 为每篇论文提供：中文标题翻译、核心亮点（一句话）、深度解析（技术方案简述）、领域归类。
    4. 输出格式如下（Markdown）：
       ### [编号]. [英文标题] (中文标题翻译)
       - **社区热度**: `👍 [此处填入对应 upvotes] Upvotes`
       - **论文链接**: [点击跳转](https://arxiv.org/abs/[此处填入对应 id])
       - **核心亮点**: ...
       - **深度解析**: ...
       - **领域归类**: [...]
       ---

    数据内容：
    {json.dumps(simple_list)}
    """

    try:
        completion = client_llm.chat.completions.create(
            model="qwen-flash", 
            messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content
        
        # 封装进折叠框
        hf_md = "<details>\n<summary><b>🤗 Hugging Face Community Choice (点击展开今日热门详情)</b></summary>\n\n"
        hf_md += "## 🤗 Hugging Face Community Choice\n\n"
        hf_md += content
        hf_md += "\n</details>"
        return hf_md
    except Exception as e:
        print(f"AI Process HF Error: {e}")
        return ""
        
def scrape_arxiv(category):
    """抓取 Arxiv 数据，并提取日期前缀和总论文数"""
    url = f"https://arxiv.org/list/{category}/recent?show=500"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        dls = soup.find_all('dl', id='articles')
        if not dls: return None, 0, []
        
        # 提取标题日期，例如：Wed, 18 Mar 2026 (showing 69 of 69 entries )
        raw_date_str = soup.find_all('h3')[0].text.strip()
        match = re.search(r'^(.*)\(showing \d+ of (\d+) entries', raw_date_str)
        if match:
            date_prefix = match.group(1).strip()
            total_entries = match.group(2)
        else:
            date_prefix = raw_date_str
            total_entries = "0"

        papers = []
        dt_tags = dls[0].find_all('dt')
        dd_tags = dls[0].find_all('dd')
        
        for dt, dd in zip(dt_tags, dd_tags):
            link_tag = dt.find('a', title='Abstract')
            if not link_tag: continue
            id_str = link_tag.text.replace('arXiv:', '').strip()
            title = dd.find('div', class_='list-title').text.replace('Title:', '').strip()
            abstract = dd.find('p', class_='mathjax').text.strip() if dd.find('p', class_='mathjax') else ""
            papers.append({"id": id_str, "title": title, "abstract": abstract[:1000]})
        
        return {"prefix": date_prefix, "total": total_entries}, len(papers), papers
    except Exception: return None, 0, []

def process_with_ai(papers):
    """恢复为直接输出 Markdown 的版本，去掉排序和评分"""
    if not papers: return ""
    global_id_counter = 1
    final_res = []
    
    for i in range(0, len(papers), 40):
        chunk = papers[i:i+40]
        prompt = f"""你是一个专注于【大模型具身智能】的顶级研究员。请从以下论文中筛选并编号。

        ✅ 必须保留：
        1. VLA (Vision-Language-Action)、World Models (世界模型)、World Modeling、视频生成。
        2. World-Action Model、Video-Action Model、Diffusion Policy。
        3. 具身 Scaling Laws、跨具身数据集、多模态融合注意力。
        
        🛑 需要剔除：
        1. 经典控制(PID, MPC)、硬件/软体/步态研究。
        2. 传统导航、路径规划(A*, RRT)、SLAM、传感器标定。
        3. 垂直场景：深海、巡检、无人机/车、攀爬、医疗/手术机器人。
        4. 经典视觉：单纯的人体姿态识别、纯 3D 重建(NeRF/GS)、单纯触觉。
        5. 多智能体协同/集群 (Swarm)、离散任务调度。

        ⚠️ 输出极其严格限制：
        1. **禁止**输出任何开场白和总结。
        2. **仅输出**符合以下格式的论文列表。
        3. **重要**：每篇论文结束后必须紧跟一个 --- 作为分割线。

        要求：
        1. 请从编号 {global_id_counter} 开始连续编号。
        2. 格式（Markdown）：
           ### {global_id_counter}. 🔥 [英文题目] (中文题目翻译)
           - **论文链接**: [点击跳转](https://arxiv.org/abs/{{id}})
           - **核心亮点**: (一句话创新点)。
           - **深度解析**: (一段话技术方案、训练规模及物理意义解析)。
           - **领域归类**: [归类版块]
           ---

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
                final_res.append(res.strip())
                global_id_counter += res.count("###")
        except Exception: pass
            
    return "\n\n".join(final_res)

def generate_archive_and_index(date_info, arxiv_content, hf_content=""):
    """生成详情页并更新索引，标题分开统计，且索引页按日期倒序排列"""
    
    # 1. 分别统计数量
    hf_count = hf_content.count("###")
    vla_count = (arxiv_content or "").count("###")
    
    # 2. 构造详情页标题 (例如: Thu, 19 Mar 2026 (HF: 10, VLA: 6 of 48 entries))
    display_title = f"{date_info['prefix']} (HF: {hf_count}, VLA: {vla_count} of {date_info['total']} entries)"
    
    safe_date_filename = re.sub(r'[^\w\s-]', '', date_info['prefix']).replace(' ', '_')
    os.makedirs('archive', exist_ok=True)
    daily_file_path = f"archive/{safe_date_filename}.html"

    # 3. 提取 Arxiv ID 用于 Sources 集合区
    paper_ids = re.findall(r'abs/(\d+\.\d+)', arxiv_content)
    sources_text = "\n".join([f"https://arxiv.org/html/{pid}" for pid in paper_ids])

    # --- HTML 模板定义 ---
    def get_html_template(title, body_content, is_index_page=False, sources_block=""):
        back_link = "<a href='../index.html' style='margin-bottom:20px; display:block;'>← 返回主索引</a>" if not is_index_page else ""
        safe_body = body_content.replace('</script>', '<\\/script>')
        return f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>{title}</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <style>
                html, body {{ -webkit-text-size-adjust: 100% !important; text-size-adjust: 100% !important; }}
                .markdown-body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
                @media (max-width: 767px) {{ .markdown-body {{ padding: 15px; }} }}
                .sources-box {{ margin-top: 50px; padding: 20px; background: #f6f8fa; border: 1px dashed #d0d7de; border-radius: 10px; }}
                .sources-box textarea {{ width: 100%; height: 100px; margin: 10px 0; padding: 10px; font-family: monospace; font-size: 12px; border: 1px solid #d0d7de; border-radius: 6px; resize: none; }}
                .copy-btn {{ background-color: #2da44e; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; }}
                details {{ margin-bottom: 20px; padding: 15px; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; }}
                summary {{ cursor: pointer; font-size: 16px; font-weight: bold; color: #0969da; }}
            </style>
        </head>
        <body class="markdown-body">
            {back_link}
            <h1>{title}</h1>
            <div id="content"></div>
            {sources_block}
            <script type="text/markdown" id="raw-markdown">{safe_body}</script>
            <script>
                const rawMdElement = document.getElementById('raw-markdown');
                if (rawMdElement) {{ document.getElementById('content').innerHTML = marked.parse(rawMdElement.textContent); }}
                function copySources() {{
                    const textArea = document.getElementById('sources-text');
                    textArea.select();
                    document.execCommand('copy');
                    alert('已复制链接，请前往 NotebookLM 粘贴');
                }}
            </script>
        </body>
        </html>
        """

    sources_html = ""
    if sources_text:
        sources_html = f"""
        <div class="sources-box">
            <h3>🔗 NotebookLM Sources 集合区 (仅 VLA 筛选共 {len(paper_ids)} 篇)</h3>
            <textarea id="sources-text" readonly>{sources_text}</textarea>
            <button class="copy-btn" onclick="copySources()">📋 复制所有来源链接</button>
        </div>
        """

    # 保存今日详情页
    full_display_content = hf_content + "\n\n" + (arxiv_content or "")
    with open(daily_file_path, "w", encoding="utf-8") as f:
        f.write(get_html_template(f"🤖 具身大模型简报 - {display_title}", full_display_content, False, sources_html))

    # --- 关键修改：优化索引页排序 ---
    history_files = [f for f in os.listdir('archive') if f.endswith('.html')]
    
    # 按照文件修改时间排序（最新创建的在最前），这样即使文件名不规范也能准确排序
    history_files.sort(key=lambda x: os.path.getmtime(os.path.join('archive', x)), reverse=True)

    index_md = "### 📅 历史存档列表\n\n"
    for f_name in history_files:
        try:
            with open(f"archive/{f_name}", "r", encoding="utf-8") as hf:
                file_soup = BeautifulSoup(hf.read(), 'html.parser')
                full_title = file_soup.title.string.replace("🤖 具身大模型简报 - ", "")
                index_md += f"- [{full_title}](archive/{f_name})\n"
        except:
            display_date_item = f_name.replace('.html', '').replace('_', ' ')
            index_md += f"- [{display_date_item}](archive/{f_name})\n"
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(get_html_template("📚 具身大模型科研日报 - 历史索引", index_md, True))

    # 推送飞书
    requests.post(FEISHU_WEBHOOK, json={
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"🌟 具身精选 | {display_title}"}, "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"今日共包含 **{hf_count}** 篇 HF 热门及 **{vla_count}** 篇 VLA 筛选论文。"}},
                {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "🌐 查看网页 & 复制 Notebook 链接"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
            ]
        }
    })

if __name__ == "__main__":
    all_p = {}
    date_info = None
    for cat in CATEGORIES:
        info, total_len, ps = scrape_arxiv(cat)
        if info: date_info = info
        for p in ps: all_p[p['id']] = p
    
    # 1. 获取 AI 筛选后的 Arxiv 内容
    arxiv_content = process_with_ai(list(all_p.values()))
    
    # 2. 处理 Hugging Face 部分 (全量总结)
    hf_raw_data = scrape_hf_daily()
    hf_content = process_hf_with_ai(hf_raw_data)
    
    # 3. 传入两个部分进行页面生成
    if date_info: 
        generate_archive_and_index(date_info, arxiv_content or "", hf_content)
