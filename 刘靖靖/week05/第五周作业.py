import math
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_s
from torch.utils.data import Dataset, DataLoader

def load_corpus(pattern="*.txt"):
    text = []
    for corpus in glob.glob(pattern):
        with open(corpus, encoding="utf-8", errors="ignore") as f:
            text.append(f.read())
    return "".join(text)

def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c:i for i, c in enumerate(chars)}
    idx2char = {i:c for c, i in char2idx.items()}
    return char2idx, idx2char

class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(batch_size, -1, self.hidden_dim)

        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        normed_x = self.norm1(x)
        attn_out = self.self_attn(normed_x, normed_x, normed_x, mask)
        x = x + self.dropout(attn_out)

        normed_x = self.norm2(x)
        ff_out = self.ff(normed_x)
        x = x + ff_out

        return x

class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = Transformer(hidden_dim, num_heads, d_ff, num_layers, dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mask=None):
        e = self.drop(self.embedding(x))
        out = self.transformer(e, mask)
        return self.fc(out)

    @torch.no_grad()
    def beam_search(self, stard_ids, max_len, beam_width=3, temperature=1.0, device=None):
        self.eval()
        if isinstance(stard_ids, list):
            stard_ids = torch.tensor(stard_ids, dtype=torch.long, device=device)
        stard_ids = stard_ids.unsqueeze(0)
        beams = [stard_ids.clone()]
        beam_scores = torch.tensor([0.0], device=device)

        for step in range(stard_ids.size(1), max_len):
            if len(beams) == 0:
                break
            all_candidates = []
            for beam_seq, score in zip(beams, beam_scores):
                logits = self.forward(beam_seq)
                last_logits = logits[:, -1, :] / temperature
                log_probs = F.log_softmax(last_logits, dim=-1)
                topk_log_prob, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                for i in range(beam_width):
                    candidate_score = score + topk_log_prob[:, i].item()
                    candidate_seq = torch.cat([beam_seq, topk_indices[:, i:i+1]], dim=1)
                    all_candidates.append((candidate_score, candidate_seq))

            all_candidates.sort(key=lambda x : x[0], reverse=True)
            beams = [seq for _, seq in all_candidates[:beam_width]]
            beam_scores = torch.tensor([score for score, _ in all_candidates[:beam_width]], device=device)
        
        return [seq.squeeze(0).tolist() for seq in beams]

    @torch.no_grad()
    def generate_with_top_p(self, start_ids, max_len, temperature=1.0, top_p=0.9, device=None):
        self.eval()
        if isinstance(start_ids, list):
            start_ids = torch.tensor(start_ids, dtype=torch.long, device=device)
        start_ids = start_ids.unsqueeze(0)
        generated = start_ids.clone()

        for step in range(start_ids.size(1), max_len):
            logits = self.forward(generated)
            last_logits = logits[:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsun(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False

            mask = sorted_mask.scatter(dim=1, index=sorted_indices, src=sorted_mask)
            last_logits[mask] = float("-inf")

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            # if next_token.item() == eos_id : break

        return generated.squeeze(0).tolist()

def run_epoch(model, loader, criterion, optimizer, device, mask=None, train=True):
    model.train(train)
    total_loss = 0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--path", default="*.txt")
    parser.add_argument("--save", default="SOTA_model.pt")
    arg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}, model:{arg.model}")

    text = load_corpus(arg.path)
    if not text:
        raise FileNotFoundError("文件夹中找不到任何.txt文件，请确认路径是否正确")
    print(f"语料库大小：{len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小：{vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - arg.eval_ratio))
    train_text = "\n".join(lines[:split])
    eval_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, arg.seq_len)
    eval_ds = CharDataset(eval_text, char2idx, arg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=arg.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=arg.batch_size, drop_last=True)

    model = LM(
        vocab_size,
        embed_dim=arg.embed_dim,
        hidden_dim=arg.hidden_dim,
        num_layers=arg.num_layers,
        num_heads=arg.num_heads,
        d_ff=arg.d_ff
    ).to(device)

    total_param = sum(p.numel() for p in model.parameters())
    print(f"总参数：{total_param:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    reduce_on_pla = lr_s.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_eval_ppl = float("inf")

    mask = torch.tril(torch.ones(arg.seq_len, arg.seq_len)).bool()
    mask = mask.unsqueeze(0).unsqueeze(0).to(device)

    print(f"\n{'EPOCH':>6} {'TRAIN_LOSS':>10} {'TRAIN_PPL':>10} {'EVAL_LOSS':>10} {'EVAL_PPL':>10}")
    print("-" * 56)

    for epoch in range(1, arg.epochs + 1):
        train_loss, train_ppl = run_epoch(model, train_loader, criterion, optimizer, device, mask, train=True)
        with torch.no_grad():
            eval_loss, eval_ppl = run_epoch(model, eval_loader, criterion, optimizer, device, mask, train=False)

        marker = " *" if best_eval_ppl > eval_ppl else ""
        if best_eval_ppl > eval_ppl:
            best_eval_ppl = eval_ppl
            torch.save({
                "model" : model.state_dict(),
                "char2idx" : char2idx,
                "idx2char" : idx2char,
                "arg" : vars(arg)
            }, arg.save)

        reduce_on_pla.step(eval_loss)

        print(f"{epoch:>6} {train_loss:>10f} {train_ppl:>10f} {eval_loss:>10f} {eval_ppl:>10f} {marker}")

    print(f"\n 训练完成，最佳PPL：{best_eval_ppl:.4f}, 已保存至：{arg.save}")

def predict(model_path, top_p, device=None):
    checkpoint = torch.load(model_path, map_location=device or "cpu")
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    arg = checkpoint["arg"]

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=arg["embed_dim"],
        hidden_dim=arg["hidden_dim"],
        num_layers=arg["num_layers"],
        num_heads=arg["num_heads"],
        d_ff=arg["d_ff"],
        dropout=0.0
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    start_text = "从前有一天，"
    start_ids = [char2idx.get(c, 0) for c in start_text]

    if top_p:
        gen_sequences = model.generate_with_top_p(start_ids, max_len=50, temperature=0.8, device=device)
    else:
        gen_sequences = model.beam_search(start_ids, max_len=50, beam_width=3, temperature=0.8, device=device)

    for i, ids in enumerate(gen_sequences):
        text = ''.join([idx2char[i] for i in ids])
        print(f"{text}")

if __name__ == '__main__':
    main()
    predict("SOTA_model.pt")
