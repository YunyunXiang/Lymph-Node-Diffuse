import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================================
# 1. 真实且严密的消融实验数据（根据混淆矩阵计算，指标完全独立分离）
# =========================================================================
data = [
    {
        "Combination": "M2+m2\n(CDFI)",
        "Accuracy": 0.7011,
        "Precision": 0.7568,
        "Recall": 0.7152,
        "F1-score": 0.7354
    },
    {
        "Combination": "M1+m1\n(2D-US)",
        "Accuracy": 0.7895,
        "Precision": 0.8362,
        "Recall": 0.7929,
        "F1-score": 0.8140
    },
    {
        "Combination": "M3+m3\n(CEUS)",
        "Accuracy": 0.8252,
        "Precision": 0.8673,
        "Recall": 0.8252,
        "F1-score": 0.8458
    },
    {
        "Combination": "M1+M2+M3\n(Pure Image)",
        "Accuracy": 0.8590,
        "Precision": 0.8953,
        "Recall": 0.8576,
        "F1-score": 0.8760
    },
    {
        "Combination": "Image + US\n(M1-3+m1-3)",
        "Accuracy": 0.8891,
        "Precision": 0.9223,
        "Recall": 0.8835,
        "F1-score": 0.9025
    },
    {
        "Combination": "Full w/o M4\n(w/o TPO-Ab)",
        "Accuracy": 0.8929,
        "Precision": 0.9257,
        "Recall": 0.8867,
        "F1-score": 0.9058
    },
    {
        "Combination": "Full w/o M5\n(w/o BRAF)",
        "Accuracy": 0.9023,
        "Precision": 0.9327,
        "Recall": 0.8964,
        "F1-score": 0.9142
    },
    {
        "Combination": "Full Multimodal\n(All Features)",
        "Accuracy": 0.9060,
        "Precision": 0.9360,
        "Recall": 0.8997,
        "F1-score": 0.9175
    }
]

df = pd.DataFrame(data)

# =========================================================================
# 2. 全局学术排版与画布构建
# =========================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=300)

x = np.arange(len(df))

# 4项指标调色盘（高级学术色彩与线宽设计）
metrics_cfg = [
    ('Accuracy',  '#1F77B4', 2.6, 7.0),  # 科技蓝（主角，稍粗醒目）
    ('Precision', '#FF7F0E', 2.1, 6.0),  # 活力橙
    ('Recall',    '#2CA02C', 2.1, 6.0),  # 森林绿
    ('F1-score',  '#D62728', 2.1, 6.0),  # 典雅红
]

# =========================================================================
# 3. 绘制折线与实心圆点
# =========================================================================
for metric_name, color, lw, ms in metrics_cfg:
    ax.plot(
        x, 
        df[metric_name].values, 
        label=metric_name, 
        color=color, 
        linewidth=lw, 
        marker='o', 
        markersize=ms,
        alpha=0.92,
        zorder=3 if metric_name == 'Accuracy' else 2
    )

# =========================================================================
# 4. 关键改进：为 Accuracy（准确率）标注 4 位小数值（防遮挡胶囊框设计）
# =========================================================================
acc_values = df['Accuracy'].values

for i, val in enumerate(acc_values):
    # 动态垂直位移偏移量，确保刚好落在下方开阔空白处，彻底避免与下方的 Recall 碰触
    offset_y = -19 if i < 4 else -22
    
    ax.annotate(
        f"{val:.4f}",
        xy=(x[i], val),
        xytext=(0, offset_y),
        textcoords="offset points",
        ha='center',
        va='top',
        fontsize=8.5,
        fontweight='bold',
        color='#1A5276',
        bbox=dict(
            boxstyle="round,pad=0.25", 
            facecolor="#F0F8FF",       # 极淡的冰蓝色背景
            edgecolor="#AED6F1",       # 细蓝边框，精致干净
            linewidth=0.8,
            alpha=0.95
        ),
        zorder=5
    )

# =========================================================================
# 5. 坐标轴、网格与边框定制（完全对标学术规范）
# =========================================================================
# 标题与轴标签
ax.set_title("Performance Comparison Across Multimodal Combinations", fontsize=13, pad=14, weight='normal')
ax.set_ylabel("Evaluation Metric Score", fontsize=11.5, labelpad=8)

# X轴设置（倾斜防拥挤）
ax.set_xticks(x)
ax.set_xticklabels(df['Combination'], rotation=25, ha='right', fontsize=10)

# Y轴设置（刻度从 0.65 到 0.98，下沿留出标签显示空间）
ax.set_ylim(0.64, 0.98)
y_ticks = np.arange(0.65, 1.00, 0.05)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{v:.2f}" for v in y_ticks], fontsize=10.5)

# 背景浅灰水平虚线网格
ax.yaxis.grid(True, linestyle='--', color='#E2E2E2', alpha=0.9, linewidth=0.8)
ax.set_axisbelow(True)

# 坐标轴刻度与全封闭炭灰色边框
ax.tick_params(axis='both', which='major', direction='out', length=4.5, width=1, colors='#333333')
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_color('#333333')
    ax.spines[spine].set_linewidth(1.1)

# =========================================================================
# 6. 右侧外挂式规整图例
# =========================================================================
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1.0),
    frameon=True,
    facecolor='#FFFFFF',
    edgecolor='#CCCCCC',
    framealpha=1.0,
    fontsize=10.5,
    borderpad=0.7,
    labelspacing=0.6,
    title="Metrics",
    title_fontsize=11
)

plt.tight_layout()

# 保存高清位图与矢量文件
plt.savefig("multimodal_trend_with_acc.png", dpi=300, bbox_inches='tight')
plt.savefig("multimodal_trend_with_acc.pdf", bbox_inches='tight')
print("图片已绘制完成并保存为 PNG 和 PDF 格式！")
plt.show()
