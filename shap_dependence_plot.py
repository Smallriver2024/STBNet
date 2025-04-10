import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import shutil

# 创建输出文件夹
output_dir = "shap_visualization"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # 删除旧文件夹以确保干净的输出
os.makedirs(output_dir)

# 创建子文件夹
dependency_dir = os.path.join(output_dir, "dependency_plots")
summary_dir = os.path.join(output_dir, "summary_plots")
interaction_dir = os.path.join(output_dir, "interaction_plots")
os.makedirs(dependency_dir)
os.makedirs(summary_dir)
os.makedirs(interaction_dir)

# 读取数据
shap_values = pd.read_csv("shap_values.csv")
train_data = pd.read_excel("train.xlsx")

# 获取变量名称（不包括Group列）
feature_names = train_data.columns.tolist()[1:]

# 确保我们有8个变量
if len(feature_names) != 8:
    print(f"警告：找到{len(feature_names)}个变量，而不是预期的8个变量")

# 配置全局图形设置，适合SCI论文发表并增大字体
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,                  # 基础字体大小增大
    'axes.titlesize': 20,             # 增大标题字体
    'axes.labelsize': 18,             # 增大轴标签字体
    'xtick.labelsize': 16,            # 增大X轴刻度字体
    'ytick.labelsize': 16,            # 增大Y轴刻度字体
    'legend.fontsize': 16,            # 增大图例字体
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.figsize': [8, 6],
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.5             # 增加轴线宽度
})

# 设置统一的散点颜色
scatter_color = '#4575b4'  # 使用蓝色作为统一颜色

# 1. 为每个变量创建美化的依赖图
for i, feature in enumerate(feature_names):
    # 提取该特征的SHAP值
    feature_shap = shap_values[feature].values
    # 获取特征的原始值
    feature_values = train_data[feature].values
    
    # 创建散点图
    plt.figure(figsize=(6, 6))
    
    # 计算散点图的点大小（基于数据集大小调整）
    point_size = max(40, min(180, 6000 / len(feature_values)))
    
    # 绘制散点图 - 使用统一颜色
    plt.scatter(feature_values, feature_shap, 
               color=scatter_color, 
               s=point_size, 
               alpha=0.8,
               edgecolor='none')
    
    # 添加水平线表示SHAP值为0
    plt.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.7)
    
    # 添加标题和标签
    plt.title(f'SHAP Dependency Plot: {feature}', fontweight='bold', pad=15, fontsize=20)
    plt.xlabel(f'{feature} Value', fontweight='bold', labelpad=15, fontsize=18)
    plt.ylabel('SHAP Value', fontweight='bold', labelpad=15, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存单独的图
    plt.savefig(f"{dependency_dir}/{feature}_dependency.pdf")
    plt.close()
    
    print(f"保存 {feature} 的依赖图")

# 2. 创建拼图布局 - 调整为更宽更一致的比例
fig = plt.figure(figsize=(20, 10))  # 进一步增加尺寸
gs = GridSpec(2, 4, figure=fig, wspace=0.28, hspace=0.4)  # 增加间距

for i, feature in enumerate(feature_names):
    # 计算行和列位置
    row = i // 4
    col = i % 4
    
    # 添加子图
    ax = fig.add_subplot(gs[row, col])
    
    # 提取该特征的SHAP值
    feature_shap = shap_values[feature].values
    # 获取特征的原始值
    feature_values = train_data[feature].values
    
    # 调整点大小
    point_size = max(35, min(150, 4000 / len(feature_values)))
    
    # 绘制散点图 - 使用统一颜色
    ax.scatter(feature_values, feature_shap, 
               color=scatter_color, 
               s=point_size, 
               alpha=0.7,
               edgecolor='none')
    
    # 添加水平线表示SHAP值为0
    ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.7)
    
    # 添加标题和标签
    ax.set_title(f'{feature}', fontweight='bold', pad=12, fontsize=20)
    ax.set_xlabel(f'{feature} Value', fontweight='bold', fontsize=18, labelpad=12)
    ax.set_ylabel('SHAP Value', fontweight='bold', fontsize=18, labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3)

# 设置整体标题
fig.suptitle('SHAP Dependency Plots for All Features', fontsize=20, fontweight='bold', y=0.98)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间

# 保存拼图
plt.savefig(f"{dependency_dir}/all_features_dependency.pdf")
plt.close()

print(f"拼图已保存到 {dependency_dir}/all_features_dependency.pdf")

# 3. 创建特征重要性汇总图（横向条形图）
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean Absolute SHAP': [np.mean(np.abs(shap_values[feature].values)) for feature in feature_names]
})
feature_importance = feature_importance.sort_values('Mean Absolute SHAP', ascending=True)

plt.figure(figsize=(11, 7))
bars = plt.barh(feature_importance['Feature'], feature_importance['Mean Absolute SHAP'], 
        color=sns.color_palette("Blues_d", len(feature_names)))

# 为每个条形添加值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=14)

plt.xlabel('Mean |SHAP Value|', fontweight='bold', labelpad=12)
plt.ylabel('Features', fontweight='bold', labelpad=12)
plt.title('Feature Importance Based on SHAP Values', fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{summary_dir}/feature_importance_ranking.pdf")
plt.close()

# 4. 创建特征方向分析图（显示正负影响）
feature_direction = pd.DataFrame({
    'Feature': feature_names,
    'Mean SHAP': [np.mean(shap_values[feature].values) for feature in feature_names]
})
feature_direction = feature_direction.sort_values('Mean SHAP')

# 创建颜色列表，负值为蓝色，正值为红色
colors = ['#4575b4' if x < 0 else '#d73027' for x in feature_direction['Mean SHAP']]

plt.figure(figsize=(11, 7))
bars = plt.barh(feature_direction['Feature'], feature_direction['Mean SHAP'], color=colors)

# 为每个条形添加值标签
for bar in bars:
    width = bar.get_width()
    plt.text(width + (0.01 if width >= 0 else -0.01), 
             bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', 
             ha='left' if width >= 0 else 'right', 
             va='center', fontweight='bold', fontsize=14)

plt.axvline(x=0, color='black', linestyle='-', linewidth=1.0)
plt.xlabel('Mean SHAP Value (Direction of Impact)', fontweight='bold', labelpad=12)
plt.ylabel('Features', fontweight='bold', labelpad=12)
plt.title('Feature Impact Direction Based on SHAP Values', fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{summary_dir}/feature_direction_analysis.pdf")
plt.close()

# 5. 创建特征组合热力图
# 计算每对特征的SHAP值的相关性
shap_corr = shap_values.corr()

plt.figure(figsize=(11, 9))
sns.heatmap(shap_corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, 
            fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Correlation'}, annot_kws={"size": 14})
plt.title('Feature SHAP Value Correlation Heatmap', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{interaction_dir}/shap_correlation_heatmap.pdf")
plt.close()

# 6. 创建双变量交互图（对每对高相关特征）
# 找出相关性最高的几对特征
corr_pairs = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr_value = abs(shap_corr.iloc[i, j])
        if corr_value > 0.3:  # 相关性阈值
            corr_pairs.append((feature_names[i], feature_names[j], corr_value))

# 按相关性排序
corr_pairs.sort(key=lambda x: x[2], reverse=True)

# 为最高相关的3对特征创建交互图
for pair_idx, (feature1, feature2, corr) in enumerate(corr_pairs[:min(3, len(corr_pairs))]):
    plt.figure(figsize=(10, 8))
    
    x = train_data[feature1].values
    y = train_data[feature2].values
    
    # 绘制散点图 - 使用统一颜色
    plt.scatter(x, y, color=scatter_color, s=100, alpha=0.7)
    
    plt.xlabel(feature1, fontweight='bold', labelpad=12)
    plt.ylabel(feature2, fontweight='bold', labelpad=12)
    plt.title(f'Feature Interaction: {feature1} vs {feature2}\nCorrelation: {corr:.3f}', 
              fontweight='bold', pad=15)
    
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{interaction_dir}/interaction_{feature1}_{feature2}.pdf")
    plt.close()

# 7. 创建Group分析图 (如果Group列包含分类数据)
if 'Group' in train_data.columns:
    groups = train_data['Group'].unique()
    
    if len(groups) <= 10:  # 只在组数量合理时创建
        # 为每个特征创建按组分组的箱形图
        for feature in feature_names:
            plt.figure(figsize=(12, 8))
            
            # 准备按组分组的数据
            data_to_plot = []
            labels = []
            
            for group in groups:
                group_indices = train_data['Group'] == group
                if sum(group_indices) > 0:  # 确保有数据
                    data_to_plot.append(shap_values.loc[group_indices, feature].values)
                    labels.append(f'Group {group}')
            
            # 创建箱形图
            box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)
            
            # 为箱子添加颜色
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(data_to_plot)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            # 添加零线
            plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
            
            plt.title(f'SHAP Value Distribution by Group: {feature}', fontweight='bold', pad=15)
            plt.ylabel('SHAP Value', fontweight='bold', labelpad=12)
            plt.xlabel('Group', fontweight='bold', labelpad=12)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{summary_dir}/group_analysis_{feature}.pdf")
            plt.close()

# 8. 创建特征分布与SHAP值关系图（带边际分布）
for feature in feature_names:
    # 创建有三个子图的布局：主图、上方直方图、右侧直方图
    fig = plt.figure(figsize=(10, 10))
    
    # 创建网格规格
    gs = GridSpec(4, 4, figure=fig)
    
    # 主图占据2-4行、0-3列
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # 上方直方图占据0-1行、0-3列
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # 右侧直方图占据2-4行、3列
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # 绘制主图（散点图）
    ax_main.scatter(train_data[feature], shap_values[feature], 
                color=scatter_color,
                s=90, alpha=0.7)
    ax_main.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.7)
    ax_main.set_xlabel(f'{feature} Value', fontweight='bold', fontsize=18, labelpad=15)
    ax_main.set_ylabel('SHAP Value', fontweight='bold', fontsize=18, labelpad=15)
    ax_main.tick_params(axis='both', which='major', labelsize=16)
    ax_main.grid(True, linestyle='--', alpha=0.3)
    
    # 绘制上方直方图（X轴数据分布）
    sns.histplot(train_data[feature], kde=True, ax=ax_top, color='#4575b4', alpha=0.7)
    # 隐藏上方直方图的各种元素
    ax_top.set_xlabel('')
    ax_top.set_ylabel('')
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_top.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # 隐藏边框
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    
    # 绘制右侧直方图（Y轴数据分布）
    sns.histplot(y=shap_values[feature], kde=True, ax=ax_right, color='#4575b4', alpha=0.7)
    # 隐藏右侧直方图的各种元素
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    ax_right.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # 隐藏边框
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(f"{dependency_dir}/{feature}_distribution_shap.pdf")
    plt.close()

# 9. 创建SHAP摘要图
plt.figure(figsize=(13, 9))

# 创建一个包含所有SHAP值的数据框
all_shap_df = pd.DataFrame()
for feature in feature_names:
    feature_df = pd.DataFrame({
        'Feature': feature,
        'SHAP Value': shap_values[feature],
        'Feature Value': train_data[feature]
    })
    all_shap_df = pd.concat([all_shap_df, feature_df])

# 按特征重要性排序
feature_order = feature_importance['Feature'].tolist()

# 创建摘要图
for i, feature in enumerate(feature_order):
    # 获取当前特征的数据
    feature_data = all_shap_df[all_shap_df['Feature'] == feature]
    
    # 绘制点 - 使用统一颜色
    plt.scatter(feature_data['SHAP Value'], i + np.random.normal(0, 0.1, len(feature_data)),
                color=scatter_color, s=70, alpha=0.7)

plt.yticks(range(len(feature_order)), feature_order, fontsize=14)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.2)
plt.xlabel('SHAP Value Impact', fontweight='bold', labelpad=12)
plt.title('Summary of Feature Effects', fontweight='bold', pad=15, fontsize=18)

plt.tight_layout()
plt.savefig(f"{summary_dir}/shap_summary_plot.pdf")
plt.close()

print(f"所有图像已保存到 {output_dir} 文件夹及其子文件夹中")

# 创建分析脚本，计算每个特征的SHAP影响方向和强度
analysis_df = pd.DataFrame(columns=['Feature', 'Mean Absolute SHAP', 'Mean SHAP', 'Direction', 'Rank'])

for feature in feature_names:
    feature_shap = shap_values[feature].values
    mean_abs_shap = np.mean(np.abs(feature_shap))
    mean_shap = np.mean(feature_shap)
    direction = "Positive" if mean_shap > 0 else "Negative"
    
    analysis_df = pd.concat([analysis_df, pd.DataFrame({
        'Feature': [feature],
        'Mean Absolute SHAP': [mean_abs_shap],
        'Mean SHAP': [mean_shap],
        'Direction': [direction]
    })], ignore_index=True)

# 按影响力大小排序
analysis_df = analysis_df.sort_values(by='Mean Absolute SHAP', ascending=False)
analysis_df['Rank'] = range(1, len(feature_names) + 1)

# 保存分析结果
analysis_df.to_csv(f"{output_dir}/feature_importance_analysis.csv", index=False)
print(f"特征重要性分析已保存到 {output_dir}/feature_importance_analysis.csv")