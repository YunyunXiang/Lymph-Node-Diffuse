import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

# ==========================================
# 1. 准备数据
# ==========================================
data = {
    'Model': ['MLP', '1D-CNN', 'BiLSTM', 'LSTM', 'ResNet18', 'Transformer'],
    'Accuracy':  [0.9060, 0.9023, 0.8778, 0.8628, 0.8365, 0.8158],
    'Precision': [0.9360, 0.9269, 0.9094, 0.8986, 0.8776, 0.8625],
    'Recall':    [0.8997, 0.9029, 0.8770, 0.8608, 0.8350, 0.8123],
    'F1-score':  [0.9175, 0.9148, 0.8929, 0.8793, 0.8557, 0.8367]
}

df = pd.DataFrame(data)

# 指标名称与维度数
categories = ['Accuracy', 'Precision', 'Recall', 'F1-score']
N = len(categories)

# 计算极坐标每个轴的角度，首尾相连实现闭合
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# ==========================================
# 2. 绘图配置与风格定制
# ==========================================
# 设置高质感科研字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), dpi=300)

# 让正北方向 (12点钟方向) 作为起始点，顺时针展开
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 设置维度轴标签与刻度样式
plt.xticks(angles[:-1], categories, size=13, weight='bold', color='#2F4F4F')
ax.tick_params(axis='x', pad=15)  # 增加标签与圆周的间距

# 设置径向轴刻度（0.80 - 0.95，每隔 0.03 一个刻度）
r_ticks = [0.80, 0.83, 0.86, 0.89, 0.92, 0.95]
ax.set_rlabel_position(22.5)  # 刻度数字偏转 22.5 度，避免与射线重合
plt.yticks(
    r_ticks, 
    [f"{t:.2f}" for t in r_ticks], 
    color="#666666", 
    size=9, 
    weight='medium'
)
plt.ylim(0.79, 0.95)

# 背景网格精细化
ax.grid(True, color='#D3D3D3', linestyle='--', linewidth=0.8, alpha=0.7)
ax.spines['polar'].set_color('#B0C4DE')
ax.spines['polar'].set_linewidth(1.2)

# ==========================================
# 3. 学术调色盘（NPG/Lancet 配色）
# ==========================================
model_colors = {
    'MLP':         '#E64B35',  # 珊瑚红 (最优模型，高亮醒目)
    '1D-CNN':      '#4DBBD5',  # 天空蓝
    'BiLSTM':      '#00A087',  # 翡翠绿
    'LSTM':        '#3C5488',  # 经典海军蓝
    'ResNet18':    '#F39B7F',  # 浅杏色
    'Transformer': '#8491B4'   # 板岩灰蓝
}

line_styles = {
    'MLP':         '-',
    '1D-CNN':      '-',
    'BiLSTM':      '-',
    'LSTM':        '--',
    'ResNet18':    '-.',
    'Transformer': ':'
}

# ==========================================
# 4. 逐个模型绘制多边形
# ==========================================
for idx, row in df.iterrows():
    model_name = row['Model']
    values = row[categories].values.flatten().tolist()
    values += values[:1]  # 闭合曲线
    
    color = model_colors[model_name]
    ls = line_styles[model_name]
    
    # 针对 MLP (最优表现) 使用略粗线宽与浅色半透明填充
    if model_name == 'MLP':
        linewidth = 2.4
        alpha_fill = 0.15
        marker = 'o'
    elif model_name == '1D-CNN':
        linewidth = 2.0
        alpha_fill = 0.08
        marker = 's'
    else:
        linewidth = 1.6
        alpha_fill = 0.02
        marker = '^'

    # 绘制折线
    ax.plot(
        angles, 
        values, 
        color=color, 
        linewidth=linewidth, 
        linestyle=ls, 
        marker=marker, 
        markersize=5,
        label=model_name
    )
    # 填充半透明多边形区域
    ax.fill(angles, values, color=color, alpha=alpha_fill)

# ==========================================
# 5. 标题、图例与保存
# ==========================================
plt.title(
    "Cross-Model Performance Comparison\nAcross Multi-Evaluation Metrics", 
    size=15, 
    weight='bold', 
    color='#1A1A1A', 
    y=1.12
)

# 图例放置在右侧居中，带有圆角背景框
plt.legend(
    loc='upper left', 
    bbox_to_anchor=(1.12, 1.02), 
    frameon=True, 
    facecolor='#FFFFFF', 
    edgecolor='#CCCCCC', 
    fontsize=11, 
    title="Model Architectures",
    title_fontsize=11,
    borderpad=0.8,
    labelspacing=0.6
)

plt.tight_layout()

# 保存高清矢量图/位图
plt.savefig("model_performance_radar.png", dpi=300, bbox_inches='tight')
plt.savefig("model_performance_radar.pdf", bbox_inches='tight')  # 便于矢量排版入文
print("Radar chart has been successfully generated and saved as PNG & PDF.")
plt.show()
