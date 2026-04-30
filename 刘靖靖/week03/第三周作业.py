"""
train_position_rnn.py
中文文本位置分类：给定一个包含“你”的5字句，判断“你”在第几位（1~5）
模型：Embedding → RNN → MaxPooling → Linear → CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 超参数
SEED        = 42
N_SAMPLES   = 5000   
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5     

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────
CHAR_POOL = list(set(
    '今天天气好晴朗我你他她它吃了吗去来吧哦是的可以不错很好超级太赞喜欢满意'
    '店铺餐厅产品服务环境系统设计课程电影情节会议堵车任务超市季节作业公交车'
    '很非常特别有点比较真是觉得感觉应该已经'
    + '你我他她它的了阿呢吗吧哦哈呀哇'
))

def make_sample():
    pos = random.randint(0, MAXLEN-1)         
    chars = [random.choice(CHAR_POOL) for _ in range(MAXLEN)]
    chars[pos] = '你'
    sent = ''.join(chars)
    return sent, pos

def build_dataset(n=N_SAMPLES):
    data = [make_sample() for _ in range(n)]
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码（不变） ──────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ─── 3. Dataset ────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]          

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.long),
                torch.tensor(self.y[i], dtype=torch.long))

# ─── 4. 模型（输出5类 logits） ─────────────────────────
class PositionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 dropout=0.3, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.lstm(self.embedding(x))
        pooled = e.max(dim=1)[0]             # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)               # (B, num_classes)

# ─── 5. 评估 ───────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred   = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total

# ─── 6. 训练主流程 ─────────────────────────────────────
def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = PositionLSTM(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你今天真好',
        '好你啊不错',
        '真的你好美',
        '你很棒哦哦',
        '今天的你很好',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred  = logits.argmax(dim=1).item()
            print(f"  预测位置: 第{pred+1}位   {sent}")

if __name__ == '__main__':
    train()
