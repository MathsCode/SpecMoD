from model.qwen3_base_model import Qwen3ForCausalLM
from transformers import AutoTokenizer
import json, tqdm
from model.utils import cos_sim_storage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_questions(question_file: str, begin=None, end=None):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


dataset = 'alpaca'
model_path = f"/inspire/hdd/global_public/public_models/Qwen/Qwen3-8B/"
dataset_path = f'/inspire/hdd/project/inference-chip/xujiaming-253308120313/dataset/benchmark/{dataset}/question.jsonl'
model = Qwen3ForCausalLM.from_pretrained(model_path) 
tokenizer = AutoTokenizer.from_pretrained(model_path)
questions = load_questions(dataset_path,0, 1)
for question in tqdm.tqdm(questions):
    messages = [
        {"role": "system",
            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
    ]
    prompt = question["turns"][0]
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking = True, 
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.000001)
for i in range(36):
    data = cos_sim_storage.get(i)

    # ==========================================
    # 0. 模拟数据 (替换为你自己的真实数据!)
    # ==========================================
    # 假设你有一系列 input_sims 和 output_sims 列表
    input_sims = []
    output_sims = []
    for meta_data in data:
        for (x,y) in meta_data:
            input_sims.append(x)
            output_sims.append(y)
    # ==========================================
    print(input_sims)
    print(output_sims)

    # ==========================================
    # 1. 设置绘图风格 (Publication Style)
    # ==========================================
    # 使用 seaborn 的 whitegrid 风格，干净专业
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # 设置英文字体，防止中文乱码 (如果需要中文请换成 SimHei 等)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


    # 创建画布，设置合适的比例 (通常 4:3 或正方形)
    plt.figure(figsize=(8, 8))

    # ==========================================
    # 2. 核心绘图：散点图 + 趋势线 (regplot)
    # ==========================================
    # sns.regplot 自动绘制散点和线性回归拟合线，并带有 95% 置信区间阴影
    # ax = sns.regplot(
    #     x=input_sims,
    #     y=output_sims,
    #     fit_reg=False,
    #     color="#1f77b4", # 经典的深蓝色
    #     scatter_kws={'alpha': 0.5, 's': 30, 'edgecolor': 'w', 'linewidths': 0.5}, # 散点样式：半透明，白色描边
    #     line_kws={'color': '#d62728', 'linewidth': 2, 'label': 'Linear Fit (Trend)'} # 趋势线样式：红色
    # )
    ax = sns.scatterplot(
        x=input_sims, 
        y=output_sims, 
        color="#1f77b4", 
        alpha=0.5, 
        edgecolor='w', 
        linewidth=0.5,
        s=30
    )
    ax.grid(which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.5)


    # ==========================================
    # 3. 添加关键参考元素
    # ==========================================
    # 添加 y=x 参考线 (Identity Line)
    # 这条线表示 "完美保持"，点离这条线越近越好
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5)


    # ==========================================
    # 4. 坐标轴与标签细节调整
    # ==========================================
    # 设置坐标轴范围，稍微留点边距，聚焦在 [0, 1] 区间
    plt.xlim(0, 1) # 根据你的数据范围调整起点，终点稍微超过 1
    plt.ylim(0, 1)
    ax.tick_params(labelbottom=False, labelleft=False)

    # 添加清晰的标签 (支持 LaTeX 公式)
    # plt.xlabel("Input Cosine Similarity (at Perturbed Layer)", fontweight='bold')
    # plt.ylabel("Output Cosine Similarity (at Final Layer)", fontweight='bold')

    # 添加标题 (可选，论文里通常写在 Caption 而不是图上)
    # plt.title("Similarity Preservation Analysis", fontsize=14, pad=15)

    # 显示图例，放在合适的角落
    # plt.legend(loc='upper left', frameon=True, shadow=True)

    # 去掉顶部和右侧的边框线，更简洁
    sns.despine()

    # 优化布局，防止标签被截断
    plt.tight_layout()

    # ==========================================
    # 5. 保存与展示
    # ==========================================
    # 保存为高分辨率矢量图 (PDF) 和栅格图 (PNG)
    plt.savefig(f"./log/sim/similarity_preservation_trend_{i}.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"./log/sim/similarity_preservation_trend_{i}.png", dpi=300, bbox_inches='tight')

    print("Plot saved successfully!")
    # plt.show()


