import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import shap

# -------------------------------------------------------------
# 0. 期刊排版全局规范 (Nature / Lancet 级别，与参考代码完全一致)
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

EXCEL_PATH = r"F:\数据集\DL_段杨\深度学习表格.xlsx"
FEATURE_ROOT = r"F:\数据集\DL_段杨\图片特征"
SAVE_DIR = r"F:\数据集\DL_段杨\SHAP_Overall_Analysis1"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES = {'无弥漫': 0, '有弥漫': 1}
MODALITY_MAP = {'2D-US': '2D-US.npy', 'CDFI': 'CDFI.npy', 'CEUS': 'CEUS.npy'}

# -------------------------------------------------------------
# 1. 临床表格数据标准化解析与编码
# -------------------------------------------------------------
df_raw = pd.read_excel(EXCEL_PATH)
df_raw.columns = [c.strip() for c in df_raw.columns]

def encode_shape(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    if '规则' in s: return 0
    if '分叶' in s: return 1
    if '成角' in s: return 2
    if '毛刺' in s: return 3
    return 0

def encode_aspect_ratio(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    if '<1' in s or '＜1' in s or s == '1': return 0
    return 1

def encode_flow(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    return 1 if '有' in s or s == '1' else 0

def encode_pattern(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    return 1 if '非向心' in s else 0

def encode_intensity(val):
    if pd.isna(val): return 1
    s = str(val).strip()
    if '低' in s: return 0
    if '等' in s: return 1
    if '高' in s: return 2
    return 1

def encode_timing(val):
    if pd.isna(val): return 1
    s = str(val).strip()
    if '晚' in s: return 0
    if '同' in s: return 1
    if '早' in s: return 2
    return 1

def encode_tpo(val):
    try:
        v = int(val)
        return v if v in [1, 2, 3] else 1
    except:
        return 1

def encode_braf(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    return 1 if '突变' in s or s == '1' else 0

clinical_records = {}
col_ar = [c for c in df_raw.columns if '纵横比' in c][0]
col_tpo = [c for c in df_raw.columns if 'TPO' in c][0]
col_braf = [c for c in df_raw.columns if 'BRAF' in c][0]

for _, row in df_raw.iterrows():
    name = str(row['姓名']).strip()
    clinical_records[name] = {
        'm1_Aspect_Ratio': encode_aspect_ratio(row[col_ar]),
        'm1_Margin_Shape': encode_shape(row['形态']),
        'm2_Vascularity': encode_flow(row['血流']),
        'm3_Enhance_Pattern': encode_pattern(row['增强模式']),
        'm3_Enhance_Intensity': encode_intensity(row['增强程度']),
        'm3_Wash_In_Time': encode_timing(row['进入时间']),
        'm3_Wash_Out_Time': encode_timing(row['消退时间']),
        'M4_TPO_Ab_Level': encode_tpo(row[col_tpo]),
        'M5_BRAF_V600E': encode_braf(row[col_braf]),
    }

# -------------------------------------------------------------
# 2. 匹配影像特征与临床数据并组装矩阵
# -------------------------------------------------------------
modality_dims = {}
for mod_name, file_name in MODALITY_MAP.items():
    for label_name in CLASSES.keys():
        folder = os.path.join(FEATURE_ROOT, label_name)
        if not os.path.exists(folder): continue
        for sub in os.listdir(folder):
            fpath = os.path.join(folder, sub, file_name)
            if os.path.isfile(fpath):
                modality_dims[mod_name] = np.load(fpath).flatten().shape[0]
                break
        if mod_name in modality_dims: break

X_img_dict = {m: [] for m in MODALITY_MAP.keys()}
mask_img_dict = {m: [] for m in MODALITY_MAP.keys()}
X_clinical_list = []
y_list = []
subject_names = []

for label_name, label_val in CLASSES.items():
    class_dir = os.path.join(FEATURE_ROOT, label_name)
    if not os.path.exists(class_dir): continue
    for sub_name in os.listdir(class_dir):
        sub_dir = os.path.join(class_dir, sub_name)
        if not os.path.isdir(sub_dir): continue
        
        for mod_name, file_name in MODALITY_MAP.items():
            fpath = os.path.join(sub_dir, file_name)
            if os.path.isfile(fpath):
                X_img_dict[mod_name].append(np.load(fpath).flatten())
                mask_img_dict[mod_name].append(True)
            else:
                X_img_dict[mod_name].append(np.zeros(modality_dims[mod_name]))
                mask_img_dict[mod_name].append(False)
        
        matched_clin = clinical_records.get(sub_name, None)
        if matched_clin is None:
            default_row = [0, 0, 0, 0, 1, 1, 1, 1, 0]
        else:
            default_row = list(matched_clin.values())
        X_clinical_list.append(default_row)
        
        y_list.append(label_val)
        subject_names.append(sub_name)

y = np.array(y_list, dtype=int)
X_clinical = np.array(X_clinical_list, dtype=np.float32)
clinical_names = ['m1_Aspect_Ratio', 'm1_Margin_Shape', 'm2_Vascularity', 
                  'm3_Enhance_Pattern', 'm3_Enhance_Intensity', 'm3_Wash_In_Time', 
                  'm3_Wash_Out_Time', 'M4_TPO_Ab_Level', 'M5_BRAF_V600E']

for m in MODALITY_MAP.keys():
    X_img_dict[m] = np.array(X_img_dict[m], dtype=np.float32)
    mask_img_dict[m] = np.array(mask_img_dict[m])

train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
)
y_train, y_test = y[train_idx], y[test_idx]

# -------------------------------------------------------------
# 3. 监督筛选与特征权重配置
# -------------------------------------------------------------
K_BEST_IMG = {'CEUS': 8, '2D-US': 6, 'CDFI': 3}

X_train_blocks, X_test_blocks = [], []
all_feature_names = []
group_priors = []
cursor = 0

for mod_name in ['CEUS', '2D-US', 'CDFI']:
    k = K_BEST_IMG[mod_name]
    valid_train = mask_img_dict[mod_name][train_idx]
    
    f_stat, _ = f_classif(X_img_dict[mod_name][train_idx][valid_train], y_train[valid_train])
    f_stat = np.nan_to_num(f_stat, 0.0)
    best_idx = np.argsort(f_stat)[::-1][:k]
    
    tr_feat = X_img_dict[mod_name][train_idx][:, best_idx]
    te_feat = X_img_dict[mod_name][test_idx][:, best_idx]
    
    mean_val = np.mean(tr_feat[valid_train], axis=0)
    tr_feat[~valid_train] = mean_val
    valid_test = mask_img_dict[mod_name][test_idx]
    te_feat[~valid_test] = mean_val
    
    X_train_blocks.append(tr_feat)
    X_test_blocks.append(te_feat)
    
    alias = {'CEUS': 'M3', '2D-US': 'M1', 'CDFI': 'M2'}[mod_name]
    weight_map = {'CEUS': 4.3, '2D-US': 3.6, 'CDFI': 1.6}
    for rank_idx in range(k):
        all_feature_names.append(f"{alias}_{mod_name}-F{rank_idx+1}")
        group_priors.append(weight_map[mod_name])
    cursor += k

clin_group_map = {
    'm1': ['m1_Aspect_Ratio', 'm1_Margin_Shape'],
    'm2': ['m2_Vascularity'],
    'm3': ['m3_Enhance_Pattern', 'm3_Enhance_Intensity', 'm3_Wash_In_Time', 'm3_Wash_Out_Time'],
    'M4': ['M4_TPO_Ab_Level'],
    'M5': ['M5_BRAF_V600E']
}
clin_weight_map = {'m1': 2.3, 'm2': 1.3, 'm3': 2.2, 'M4': 2.0, 'M5': 1.1}

for grp_key, grp_cols in clin_group_map.items():
    grp_indices = [clinical_names.index(col) for col in grp_cols]
    tr_c = X_clinical[train_idx][:, grp_indices]
    te_c = X_clinical[test_idx][:, grp_indices]
    
    X_train_blocks.append(tr_c)
    X_test_blocks.append(te_c)
    
    for c_name in grp_cols:
        all_feature_names.append(c_name)
        group_priors.append(clin_weight_map[grp_key])
    cursor += len(grp_cols)

X_train_final = np.concatenate(X_train_blocks, axis=1)
X_test_final = np.concatenate(X_test_blocks, axis=1)
feature_priors = np.array(group_priors, dtype=np.float32)

# -------------------------------------------------------------
# 4. XGBoost 拟合与 SHAP 计算
# -------------------------------------------------------------
dtrain = xgb.DMatrix(X_train_final, label=y_train, feature_names=all_feature_names)
dtrain.set_info(feature_weights=feature_priors)
dtest = xgb.DMatrix(X_test_final, label=y_test, feature_names=all_feature_names)

params = {
    'max_depth': 4,
    'learning_rate': 0.035,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'alpha': 0.3,
    'lambda': 1.2,
    'seed': 42
}

bst = xgb.train(params, dtrain, num_boost_round=140)
explainer = shap.TreeExplainer(bst)
shap_explanation = explainer(dtest)
shap_explanation.feature_names = all_feature_names

# -------------------------------------------------------------
# 5. 精选 Top 17 特征切片 (彻底剔除 "Sum of other features")
# -------------------------------------------------------------
# 按特征在测试集上的全局绝对平均值降序排列，取前 17 个特征
mean_abs_impact = np.mean(np.abs(shap_explanation.values), axis=0)
top17_indices = np.argsort(mean_abs_impact)[::-1][:17]

# 重新构建仅包含该 17 个特征的 Explanation 对象
shap_explanation_top17 = shap.Explanation(
    values=shap_explanation.values[:, top17_indices],
    base_values=shap_explanation.base_values,
    data=shap_explanation.data[:, top17_indices] if shap_explanation.data is not None else None,
    feature_names=[all_feature_names[i] for i in top17_indices]
)

# -------------------------------------------------------------
# 6. 图表: 蜂巢图 (Beeswarm Plot，完全对齐参考代码尺寸与离散度)
# -------------------------------------------------------------
plt.close('all')
fig = plt.figure(figsize=(8.5, 5.5), dpi=300)

shap.plots.beeswarm(
    shap_explanation_top17, 
    max_display=17,       # 仅展示 17 个特征，由于总数就是 17，不会生成任何剩余特征汇总结算
    plot_size=None,
    alpha=0.85,           # 保持完全一致的点透明度与离散感
    show=False
)

ax = plt.gca()
ax.set_title("Global SHAP attribution of top 17 ultrasound imaging & clinical features for thyroid diffuse infiltration", 
             fontsize=11, weight='bold', pad=12)
ax.set_xlabel("SHAP Value (Impact on Log-odds for Diffuse Infiltration)", fontsize=9.5, weight='semibold')
ax.tick_params(axis='both', which='major', labelsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "Fig2_Global_SHAP_Beeswarm.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, "Fig2_Global_SHAP_Beeswarm.pdf"), format='pdf', bbox_inches='tight')
plt.show()

print(f"\n[Finished] Top 17 特征蜂巢图已成功生成并保存至目录：\n{SAVE_DIR}")
