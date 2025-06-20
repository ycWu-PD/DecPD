import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 预设的混淆矩阵，包含整数数据
cm = np.array([[200, 0, 0, 0, 0, 0, 0],
               [0, 50, 0, 0, 0, 0, 0],
               [1, 0, 41, 0, 0, 0, 0],
               [0, 0, 2, 38, 1, 1, 0],
               [0, 0, 0, 5, 32, 0, 0],
               [0, 0, 1, 0, 0, 33, 2],
               [1, 0, 0, 0, 0, 1, 39]])

# 类别标签
labels = ["IID", "SCD", "EMI", "CSMP", "OSCL", "ETF", "non-PD"]

# 将混淆矩阵转换为百分比
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 创建绘图，并设置宽度为 2.5 英寸，高度为 2.5 英寸
plt.figure(figsize=(3.4, 3.4))

# 使用 seaborn 绘制百分比混淆矩阵
ax = sns.heatmap(cm_percentage, annot=True, cmap='Blues', cbar=False, square=True,
                 annot_kws={"size": 10, "fontname": "Times New Roman"},  # 设置字体
                 linewidths=0,  # 去掉内部的线条
                 linecolor='black',  # 外侧框线的颜色
                 cbar_kws={"shrink": 0.8},  # 调整颜色条的大小
                 vmin=0, vmax=100,  # 设置颜色差异范围，增强对比度
                 xticklabels=labels, yticklabels=labels, fmt='.2f')  # 设置显示格式，避免科学记数法

# 设置刻度位置到每个格子的中间
ax.set_xticks(np.arange(cm.shape[1]) + 0.3)  # 设置x轴刻度到每个格子的中间
ax.set_yticks(np.arange(cm.shape[0]) + 0.5)  # 设置y轴刻度到每个格子的中间

# 去除刻度线（刻度的位置仍然保留）
ax.set_xticklabels(labels, fontstyle='italic', fontsize=10, fontname='Times New Roman')  # 设置斜体并保留刻度标签
ax.set_yticklabels(labels, fontstyle='italic', fontsize=10, fontname='Times New Roman', rotation=0)  # 设置斜体并保留刻度标签，旋转为0（水平显示）

# 设置外侧边框的线条宽度为1
for _, spine in ax.spines.items():
    spine.set_visible(True)  # 确保外侧边框可见
    spine.set_linewidth(1)  # 设置边框的宽度为1

# 调整刻度线的宽度
ax.tick_params(axis='both', which='major', width=0.5)  # 调整刻度线的宽度为 0.5

# 设置X轴刻度标签旋转 45°
plt.xticks(rotation=45)

# 添加“%”标注到右下角
plt.text(7.5, 7.2, '(%)', horizontalalignment='center', verticalalignment='center',
         fontsize=12, fontname='Times New Roman', fontstyle='italic', color='black', weight='bold')

# 去掉空白边距，保存为指定的 SVG 文件路径
plt.tight_layout(pad=0)  # 去掉所有空白边距
plt.savefig(r"C:\Users\86178\Desktop\论文\图\原始图\Confusion matrix under noises.svg", format="svg", bbox_inches='tight', transparent=True)

# 关闭图像（不显示）
plt.close()
