import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
import seaborn as sns

# =====================
# 1. 基本设置
# =====================
dataset_root = "/mnt/dataset4/sitian/DailyEEG_dataset"
out_dir = "/mnt/dataset4/Yuxi/image"

os.makedirs(out_dir, exist_ok=True)

emotion_cols = [
    'inspired',
    'determined',
    'attentive',
    'active',
    'alert',
    'upset',
    'hostile',
    'ashamed',
    'nervous',
    'afraid'
]


subjects = sorted(glob.glob(os.path.join(dataset_root, "sub-*")))

print("最终使用被试数量:", len(subjects))
print(subjects)
all_subject_data = []
subject_corrs = []

# =====================
# 2. 逐个被试读取 + 被试内 z-score
# =====================
for sub_path in subjects:
    subject_id = os.path.basename(sub_path)

    beh_files = sorted(
        glob.glob(
            os.path.join(
                sub_path,
                "ses-*",
                "beh",
                "*.tsv"
            )
        )
    )

    if len(beh_files) == 0:
        raise RuntimeError(
            f"[FATAL ERROR] No beh files found for subject: {subject_id}\n"
            f"Path: {sub_path}\n"
            "Pipeline stopped to prevent silent data corruption."
        )

    print(subject_id, "共有", len(beh_files), "个beh文件")

    raw_data = pd.concat(
        [pd.read_csv(f, sep="\t") for f in beh_files],
        ignore_index=True
    )

    raw_data = raw_data[
        [
            "inspired",
            "determined",
            "attentive",
            "active",
            "alert",
            "upset",
            "hostile",
            "ashamed",
            "nervous",
            "afraid"
        ]
    ]

    raw_data = raw_data.apply(pd.to_numeric, errors="coerce")

    # 去掉全是空值的行
    raw_data = raw_data.dropna(how='all')

    # 被试内 z-score
    # 每个被试、每个情绪列分别算均值和标准差
    mean = raw_data.mean()
    std = raw_data.std()

    # 防止某一列标准差为0导致除不动
    std = std.replace(0, np.nan)

    z_data = (raw_data - mean) / std
    # 检查 z-score 后有没有 NaN
    nan_count = z_data.isna().sum().sum()

    if nan_count > 0:
        print(f"\n{subject_id} 出现 {nan_count} 个 NaN")
        print(z_data.isna().sum())

    # 如果有 NaN，用 0 填掉
    # sub11 的ashamed标准差为0
    z_data = z_data.fillna(0)
    # 每个被试内部单独计算相关矩阵
    sub_corr = z_data[emotion_cols].corr()
    subject_corrs.append((subject_id, sub_corr))

    z_data["subject"] = subject_id
    all_subject_data.append(z_data)
  

# =====================
# 3. 合并所有被试
# =====================
all_data = pd.concat(all_subject_data, ignore_index=True)

print("合并后数据形状:", all_data.shape)
print(all_data.head())

# 只取情绪列做 PCA
X = all_data[emotion_cols].values

print("X shape:", X.shape)

# =====================
# 4. PCA 降维
# =====================
pca = decomposition.PCA(n_components=2)
X_new = pca.fit_transform(X)

print("X_new shape:", X_new.shape)
print("解释方差比例:", pca.explained_variance_ratio_)
# =====================
# 画每个被试在二维PCA空间中的 trajectory
# =====================

# 把 PCA 后的二维坐标放回 all_data
all_data["PC1"] = X_new[:, 0]
all_data["PC2"] = X_new[:, 1]

# 给每个被试内部加评分顺序编号
all_data["rating_order"] = all_data.groupby("subject").cumcount()

# =====================
# PCA trajectory
# =====================

subjects = sorted(
    all_data["subject"].unique(),
    key=lambda x: int(x.replace("sub-", ""))
)

n_subjects = len(subjects)
ncols = 5
nrows = int(np.ceil(n_subjects / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(20, 4 * nrows),
    sharex=False,
    sharey=False
)

axes = axes.flatten()

for idx, subject in enumerate(subjects):
    ax = axes[idx]

    sub_data = all_data[all_data["subject"] == subject].copy()
    sub_data = sub_data.sort_values("rating_order")

    ax.plot(
        sub_data["PC1"],
        sub_data["PC2"],
        marker="o",
        markersize=3,
        linewidth=1
    )

    ax.scatter(
        sub_data["PC1"].iloc[0],
        sub_data["PC2"].iloc[0],
        color="green",
        s=50,
        label="Start"
    )

    ax.scatter(
        sub_data["PC1"].iloc[-1],
        sub_data["PC2"].iloc[-1],
        color="red",
        s=50,
        label="End"
    )

    ax.set_title(subject)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(-5, 9)
    ax.set_ylim(-5, 5)
    ax.grid(True, linestyle="--", alpha=0.3)

# 多余的空白子图删掉
for idx in range(n_subjects, len(axes)):
    fig.delaxes(axes[idx])

# 只放一个总图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper right"
)

plt.suptitle(
    "PCA Trajectories of All Subjects",
    fontsize=18
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

save_path = os.path.join(
    out_dir,
    "all_subjects_pca_trajectories.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("所有被试的 PCA trajectory 大图已保存:", save_path)
# =====================
# 5. 画 PCA Loadings 图
# =====================
loadings = pca.components_

plt.figure(figsize=(9, 8))
plt.imshow(loadings, cmap='viridis', aspect='auto')
plt.colorbar(label='Loadings')
plt.title('PCA Loadings after Subject-wise Z-score')
plt.xlabel('Features')
plt.ylabel('Principal Components')
plt.xticks(
    ticks=np.arange(loadings.shape[1]),
    labels=emotion_cols,
    rotation=90
)
plt.yticks(
    ticks=np.arange(loadings.shape[0]),
    labels=np.arange(1, loadings.shape[0] + 1)
)

save_path = os.path.join(out_dir, "all_subjects_pca_loadings.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print("保存:", save_path)



#热图
# corr = all_data[emotion_cols].corr()
# corr = sum(subject_corrs) / len(subject_corrs)
corr_sum = pd.DataFrame(0.0, index=emotion_cols, columns=emotion_cols)
corr_count = pd.DataFrame(0, index=emotion_cols, columns=emotion_cols)
for subject_id, sub_corr in subject_corrs:
    for row in emotion_cols:
        for col in emotion_cols:
            if subject_id == "sub-11" and (row == "ashamed" or col == "ashamed"):
                continue

            corr_sum.loc[row, col] += sub_corr.loc[row, col]
            corr_count.loc[row, col] += 1
print("相关矩阵计数：")
print(corr_count)
corr = corr_sum / corr_count
np.fill_diagonal(corr.values, np.nan)
print("情绪相关性矩阵:\n", corr)
plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,      # 显示数字
    cmap="coolwarm", # 红蓝配色
    center=0,
    fmt=".2f",
    square=True
)

plt.title("Emotion Correlation Matrix")

save_path = os.path.join(
    out_dir,
    "emotion_correlation_heatmap.png"
)

plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print("保存:", save_path)
corr_abs = corr.abs()

pairs = []

for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        pairs.append(
            (
                corr.columns[i],
                corr.columns[j],
                corr.iloc[i,j]
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: abs(x[2]),
    reverse=True
)

print("\nTop 10 strongest correlations:")

for p in pairs[:10]:
    print(
        f"{p[0]} <-> {p[1]} : {p[2]:.3f}"
    )
    
corr_save_path = os.path.join(
    out_dir,
    "emotion_correlation_matrix.xlsx"
)

corr.to_excel(corr_save_path)

print("保存:", corr_save_path)