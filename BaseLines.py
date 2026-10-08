import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 1. 配置与路径 =================
DATA_PATH = r'F:\数据集\DL_段杨\图片特征'
EXCEL_PATH = r'F:\数据集\DL_段杨\深度学习表格_清除格式.xlsx'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================= 2. 数据处理流水线 =================

def load_and_prepare_data():
    print("正在加载数据并进行互信息特征筛选，请稍候...")
    
    # A. 读取Excel
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip() # 去除表头空格
    
    # 临床特征列名
    clinical_cols = ['纵横比', '形态', '血流', '增强模式', '增强程度', '进入时相', 'TPO-Ab', 'BRAF V600E']
    
    # 类别特征编码
    le = LabelEncoder()
    for col in clinical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    raw_img_list = []
    raw_clin_list = []
    labels = []
    
    # B. 遍历文件夹匹配数据
    # 类别：无弥漫 (0), 有弥漫 (1)
    for label, folder in enumerate(['无弥漫', '有弥漫']):
        p = os.path.join(DATA_PATH, folder)
        if not os.path.exists(p): continue
        
        for sub_name in os.listdir(p):
            sub_path = os.path.join(p, sub_name)
            # 匹配Excel姓名
            sub_info = df[df['姓名'].astype(str).str.strip() == str(sub_name).strip()]
            if sub_info.empty: continue
            
            # 提取三个.npy模态
            m_files = ['2D-US.npy', 'CDFI.npy', 'CEUS.npy']
            expected_dims = [2115, 2117, 6345]
            combined_img = []
            
            for i, f_name in enumerate(m_files):
                f_path = os.path.join(sub_path, f_name)
                if os.path.exists(f_path):
                    v = np.load(f_path).flatten()
                    v = np.nan_to_num(v)
                    # 强行对齐维度
                    if len(v) < expected_dims[i]: v = np.pad(v, (0, expected_dims[i]-len(v)))
                    else: v = v[:expected_dims[i]]
                    combined_img.append(v)
                else:
                    combined_img.append(np.zeros(expected_dims[i]))
            
            raw_img_list.append(np.concatenate(combined_img))
            raw_clin_list.append(sub_info[clinical_cols].values[0])
            labels.append(label)

    X_img_raw = np.array(raw_img_list)
    X_clin_raw = np.array(raw_clin_list)
    y = np.array(labels)

    # C. 互信息特征选择 (SelectKBest)
    # 仅对万维图像特征进行监督筛选，保留相关性最强的200维
    print(f"原始图像特征维度: {X_img_raw.shape[1]}, 开始互信息筛选...")
    selector = SelectKBest(score_func=mutual_info_classif, k=200)
    X_img_selected = selector.fit_transform(X_img_raw, y)
    
    # D. 标准化与最终合并
    scaler = StandardScaler()
    X_img_final = scaler.fit_transform(X_img_selected)
    X_clin_final = scaler.fit_transform(X_clin_raw)
    
    X_all = np.concatenate([X_img_final, X_clin_final], axis=1)
    print(f"筛选完成！最终输入维度: {X_all.shape[1]} (200图像 + 8临床)")
    
    return X_all, y

# ================= 3. 六大基线模型定义 =================

# 1. MLP
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# 2. 1D-CNN
class CNN1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, in_dim)),
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 2)
        )
    def forward(self, x): return self.net(x)

# 3. LSTM (及 4. BiLSTM 通用)
class RNNModel(nn.Module):
    def __init__(self, in_dim, bi=False):
        super().__init__()
        # 208 拆分为 13步 * 16特征
        self.rnn = nn.LSTM(16, 32, batch_first=True, bidirectional=bi)
        self.fc = nn.Linear(32 * (2 if bi else 1), 2)
    def forward(self, x):
        x = x.view(-1, 13, 16)
        _, (h, _) = self.rnn(x)
        out = torch.cat((h[-2], h[-1]), dim=1) if hasattr(self.rnn, 'bidirectional') and self.rnn.bidirectional else h[-1]
        return self.fc(out)

# 5. Transformer
class TransformerModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embed = nn.Linear(16, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        x = self.embed(x.view(-1, 13, 16))
        x = self.trans(x)
        return self.fc(x.mean(dim=1))

# 6. ResNet18-1D (精简适配版)
class ResNet18_1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.prep = nn.Sequential(nn.Unflatten(1, (1, in_dim)), nn.Conv1d(1, 64, 3, padding=1), nn.ReLU())
        self.layer = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.down = nn.Sequential(nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = self.prep(x)
        x = x + self.layer(x)
        x = self.down(x)
        return self.fc(x.mean(dim=2))

# ================= 4. 训练与五折交叉验证引擎 =================

def run_cv_experiment(model_name, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    metrics = []
    best_acc = 0
    best_cm = None
    
    print(f"\n开始训练模型: {model_name}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        tx, vx = torch.FloatTensor(X[train_idx]), torch.FloatTensor(X[val_idx])
        ty, vy = torch.LongTensor(y[train_idx]), torch.LongTensor(y[val_idx])
        
        loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=True)
        
        # 初始化模型
        in_dim = X.shape[1]
        if model_name == 'MLP': model = MLP(in_dim)
        elif model_name == '1D-CNN': model = CNN1D(in_dim)
        elif model_name == 'LSTM': model = RNNModel(in_dim, bi=False)
        elif model_name == 'BiLSTM': model = RNNModel(in_dim, bi=True)
        elif model_name == 'Transformer': model = TransformerModel(in_dim)
        elif model_name == 'ResNet18': model = ResNet18_1D(in_dim)
        
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # 针对样本不平衡给予权重 (223 vs 309)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.38, 1.0]).to(DEVICE))
        
        # 训练
        for epoch in range(100):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            outputs = model(vx.to(DEVICE))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # 计算指标
        acc = accuracy_score(vy, preds)
        pre = precision_score(vy, preds, zero_division=0)
        rec = recall_score(vy, preds, zero_division=0)
        f1 = f1_score(vy, preds, zero_division=0)
        cm = confusion_matrix(vy, preds)
        
        metrics.append([acc, pre, rec, f1])
        if acc > best_acc:
            best_acc = acc
            best_cm = cm

    return np.mean(metrics, axis=0), best_acc, best_cm

# ================= 5. 主程序运行 =================

if __name__ == "__main__":
    try:
        # 1. 加载并处理数据
        X_final, y_final = load_and_prepare_data()
        
        # 2. 遍历模型
        model_list = ['MLP', '1D-CNN', 'LSTM', 'BiLSTM', 'Transformer', 'ResNet18']
        final_results = {}

        for name in model_list:
            avg_m, b_acc, b_cm = run_cv_experiment(name, X_final, y_final)
            final_results[name] = {'avg': avg_m, 'best_acc': b_acc, 'cm': b_cm}

        # 3. 输出报表
        print("\n" + "="*70)
        print(f"{'模型名称':<15} | {'准确率':<8} | {'精确率':<8} | {'召回率':<8} | {'F1值':<8}")
        print("-" * 70)
        for name in model_list:
            m = final_results[name]['avg']
            print(f"{name:<15} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | {m[3]:.4f}")
            print(f"  > 最优折准确率: {final_results[name]['best_acc']:.4f}")
            print(f"  > 最优折混淆矩阵:\n{final_results[name]['cm']}")
            print("-" * 70)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
