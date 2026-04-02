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
# 这里的 URL 会根据你的新仓库名自动变化
GITHUB_PAGES_URL = f"https://{repo_owner}.github.io/{repo_name}/"

CATEGORIES = ['cs.RO']

def scrape_arxiv(category):
    """抓取 Arxiv 数据，并提取日期前缀和总论文数"""
    url = f"https://arxiv.org/list/{category}/recent?show=500"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        dls = soup.find_all('dl', id='articles')
        if not dls: return None, 0, []
        
        # 提取标题日期
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

def get_arxiv_full_text(paper_id):
    """利用 Arxiv HTML 渲染功能抓取正文"""
    url = f"https://arxiv.org/html/{paper_id}"
    try:
        res = requests.get(url, timeout=20)
        if res.status_code != 200: return None
        soup = BeautifulSoup(res.text, 'html.parser')
        # 移除脚本和样式，保留前 30000 字符以防超出大模型上下文
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()[:30000] 
    except Exception as e:
        print(f"抓取全文出错 {paper_id}: {e}")
        return None

def process_with_ai(papers):
    """两阶段处理：先按原始 Prompt 筛选，再读全文深度总结"""
    if not papers: return ""
    
    # --- 第一阶段：初筛 (完全使用你提供的原始 Prompt 逻辑) ---
    all_filtered_papers = []
    for i in range(0, len(papers), 40):
        chunk = papers[i:i+40]
        # 这里嵌入你提供的原始筛选 Prompt
        filter_prompt = f"""你是一个专注于【大模型具身智能】的顶级研究员。请从以下论文中筛选出符合要求的论文，并为它们打分（1-10分，10分为极度相关）。

        ✅ 必须保留（相关度 7-10 分）：
        1. VLA (Vision-Language-Action)、World Models (世界模型)、World Modeling、视频生成。
        2. World-Action Model、Video-Action Model、Diffusion Policy。
        3. 具身 Scaling Laws、跨具身数据集、多模态融合注意力。
        
        🛑 需要剔除（直接丢弃，不要输出）：
        1. 经典控制(PID, MPC)、硬件/软体/步态研究。
        2. 传统导航、路径规划(A*, RRT)、SLAM、传感器标定。
        3. 垂直场景：深海、巡检、无人机/车、攀爬、医疗/手术机器人。
        4. 经典视觉：单纯的人体姿态识别、纯 3D 重建(NeRF/GS)、单纯触觉。
        5. 多智能体协同/集群 (Swarm)、离散任务调度。

        ⚠️ 输出极其严格限制：
        请**仅输出 JSON 格式的数组**，不要包含任何其他解释文字或 Markdown 标记。
        
        待处理数据：
        {json.dumps(chunk)}
        """
        
        try:
            completion = client_llm.chat.completions.create(
                model="qwen-flash", 
                messages=[{"role": "user", "content": filter_prompt}]
            )
            res = completion.choices[0].message.content
            match = re.search(r'\[.*\]', res, re.DOTALL)
            if match:
                all_filtered_papers.extend(json.loads(match.group(0)))
        except Exception: pass

    # --- 第二阶段：针对高分论文阅读全文并进行专家解析 ---
    # 筛选出评分 >= 8 的精选论文进行深度“脱水”
    high_quality_papers = [p for p in all_filtered_papers if p.get('score', 0) >= 8]
    final_reports = []
    
    for idx, item in enumerate(high_quality_papers, 1):
        paper_id = item['id']
        full_text = get_arxiv_full_text(paper_id)
        
        # 专家深度解析 Prompt（结合你要求的“刻薄、专业”要求）
        expert_prompt = f"""
        Role: 你是一位极其挑剔、实战经验丰富的【大模型具身智能】顶级专家。你不仅关注学术指标，更关注代码层面的实现细节和物理世界的落地可能。
        Task: 请基于提供的论文内容进行深度“脱水”解析。你的回答必须保持客观、刻薄但专业，严格遵循以下结构：

        1. 核心改进定位（去包装化）
        - 一句话总结：用最直白的话说出这篇论文到底改了什么。
        - SOTA 溯源：哪些是成熟开源方案，哪些才是作者原创的“Private Sauce”？

        2. 痛点与方法的“因果映射”
        - 作者宣称的局限性：论文开头吐槽了前人哪些问题？
        - 技术补丁：针对上述每个问题，作者分别用了什么“技术补丁”？
        - 有效性质疑：这种对应关系在逻辑上是否自洽？

        3. 硬件与架构深度拆解
        - 模块参数：详细列出模型的各个组成部分，输入输出等。
        - 动作生成逻辑：Action Head 是离散 Token 还是连续扩散？是否支持 Action Chunking？
        - 数据流向闭环：数据从相机采样到推理输出，再到机械臂执行的完整链路。
        - 损失函数（Loss）：核心 Loss 是什么？

        4. 实战落地评估（工程视角）
        - 推理性能：能否支持 20Hz 以上的实时闭环控制？
        - 资源消耗：大概需要什么样的算力规模？
        - 物理意义：处理“非刚体（如揉搓衣服）”时，是否有专门的归纳偏置设计？

        5. 风险与局限（Critical Catch）
        - 实验中的“坑”：实验环境是否过于洁净？成功率下降的典型场景是什么？
        - 潜在风险：如果我把这个架构搬到我的 Franka/RealMan 实验室，最可能在哪个环节出问题？

        待处理全文内容：
        {full_text if full_text else "（全文抓取失败，请基于摘要进行尽量详尽的分析）"}
        """
        
        try:
            # 深度解析建议用逻辑更强的模型（如 qwen-plus）
            completion = client_llm.chat.completions.create(
                model="qwen-plus", 
                messages=[{"role": "user", "content": expert_prompt}]
            )
            report = completion.choices[0].message.content
            
            score = item.get('score', 0)
            md = f"### {idx}. 🔥 [{item.get('title_en', 'Unknown')}] ({item.get('title_zh', '')})\n"
            md += f"- **专家评分**: `{score}/10` | **Arxiv**: [点击跳转](https://arxiv.org/abs/{paper_id})\n"
            md += f"{report}\n"
            md += "---\n"
            final_reports.append(md)
        except Exception as e:
            print(f"深度解析出错 {paper_id}: {e}")
            
    return "\n\n".join(final_reports)

def generate_archive_and_index(date_info, arxiv_content):
    """生成详情页并更新索引，仅统计 VLA 内容"""
    
    # 只统计 Arxiv/VLA 数量
    vla_count = (arxiv_content or "").count("###")
    
    # 构造详情页标题，移除了 HF 统计
    display_title = f"{date_info['prefix']} (VLA: {vla_count} of {date_info['total']} entries)"
    
    safe_date_filename = re.sub(r'[^\w\s-]', '', date_info['prefix']).replace(' ', '_')
    os.makedirs('archive', exist_ok=True)
    daily_file_path = f"archive/{safe_date_filename}.html"

    paper_ids = re.findall(r'abs/(\d+\.\d+)', arxiv_content)
    sources_text = "\n".join([f"https://arxiv.org/html/{pid}" for pid in paper_ids])

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
                    alert('已复制链接');
                }}
            </script>
        </body>
        </html>
        """

    sources_html = ""
    if sources_text:
        sources_html = f"""
        <div class="sources-box">
            <h3>🔗 NotebookLM Sources 集合区 (共 {len(paper_ids)} 篇)</h3>
            <textarea id="sources-text" readonly>{sources_text}</textarea>
            <button class="copy-btn" onclick="copySources()">📋 复制所有来源链接</button>
        </div>
        """

    # 仅保存 Arxiv 内容
    with open(daily_file_path, "w", encoding="utf-8") as f:
        f.write(get_html_template(f"🤖 具身大模型简报 - {display_title}", arxiv_content or "", False, sources_html))

    history_files = [f for f in os.listdir('archive') if f.endswith('.html')]
    indexed_history = []
    for f_name in history_files:
        try:
            with open(f"archive/{f_name}", "r", encoding="utf-8") as hf:
                file_soup = BeautifulSoup(hf.read(), 'html.parser')
                full_title = file_soup.title.string.replace("🤖 具身大模型简报 - ", "")
                date_match = re.search(r'([A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4})', full_title)
                if date_match:
                    date_obj = datetime.datetime.strptime(date_match.group(1), "%a, %d %b %Y")
                    indexed_history.append((date_obj, full_title, f_name))
        except: continue

    indexed_history.sort(key=lambda x: x[0], reverse=True)
    index_md = "### 📅 历史存档列表\n\n"
    for _, display_title, f_name in indexed_history:
        index_md += f"- [{display_title}](archive/{f_name})\n"
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(get_html_template("📚 具身大模型科研日报 - 历史索引", index_md, True))

    # 飞书推送卡片只显示 VLA 数量
    requests.post(FEISHU_WEBHOOK, json={
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"🌟 具身精选 | {display_title}"}, "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md", "content": f"今日共包含 **{vla_count}** 篇 VLA 筛选论文。"}},
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
    
    # 删除了 Hugging Face 的抓取和处理逻辑
    arxiv_content = process_with_ai(list(all_p.values()))
    
    if date_info: 
        generate_archive_and_index(date_info, arxiv_content or "")
