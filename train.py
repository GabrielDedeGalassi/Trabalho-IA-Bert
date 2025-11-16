# train.py
"""
O código treina um classificador BERT para os seguintes labels: onca, caseiro e fake_news.
Pra executar no colab:
  !python train.py --label onca
  !python train.py --label fake_news
  !python train.py --label all
Obs: Foi alterado o nome da coluna no excel para fake_news. Para uso próprio,
sugiro que altere a coluna para fake_news.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Configurações iniciais
# --------------------------
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 10
MAX_LENGTH = 128
SEED = 42
DATA_PATH = "oncas_comentarios.xlsx"  # Mesma pasta do script
OUTPUT_DIR = "models"

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------
# Pré-processamento
# --------------------------
def prepare_dataframe_for_label(df, label):
    """
    Dado o dataframe bruto e o nome da label que queira usar ('onca','caseiro','fake_news'),
    retorna o dataframe da coluna dessa 'label' contendo valores numéricos:
      - Para onca/caseiro: mapeia positivo->2, neutro->1, negativo->0
      - Para fake_news: mapeia sim->1, não->0
    Em resumo, ele mantem apenas linhas com labels válidas e os comentários que não estão vazios.
    """
    df = df.copy()
    df = df.dropna(subset=["comment_text"])
    df = df.drop_duplicates(subset=["comment_text"])
    # Formata as colunas de texto para strings minusculas sem formatação
    if label in ["onca", "caseiro"]:
        map_sent = {"positivo": 2, "neutro": 1, "negativo": 0}
        # Formata os valores (acentos/espaços)
        df[label] = df[label].astype(str).str.strip().str.lower()
        # Se houver problemas de codificação, como 'nao' em vez de 'não', etc.,
        # Deixe como está: Serão filtradas pelas chaves em `map`.
        # Mantenha apenas as linhas em que a label corresponde a uma das chaves de `map_sent`.
        df = df[df[label].isin(map_sent.keys())]
        df = df.rename(columns={label: "label_raw"})
        df["label"] = df["label_raw"].map(map_sent)
        classes = [0,1,2]
        target_names = ["negativo","neutro","positivo"]
    elif label == "fake_news":
        map_bin = {"sim": 1, "não": 0, "nao": 0}
        df[label] = df[label].astype(str).str.strip().str.lower()
        # Formata as variantes comuns
        df[label] = df[label].replace({"n�o":"não"})
        df = df[df[label].isin(map_bin.keys())]
        df = df.rename(columns={label: "label_raw"})
        df["label"] = df["label_raw"].map(map_bin)
        classes = [0,1]
        target_names = ["não","sim"]
    else:
        raise ValueError("Unknown label")
    df = df.reset_index(drop=True)
    return df, classes, target_names

# --------------------------
# Classe do Dataset
# --------------------------
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --------------------------
# Modelo (usa token CLS)
# --------------------------
class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Embedding do CLS
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return self.classifier(x)

# --------------------------
# Funções de Treino e Avaliação
# --------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def test_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(ids, mask)
            p = outputs.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p)
            trues.extend(labels.cpu().numpy().tolist())
    return trues, preds

# --------------------------
# Execução de experimento único
# --------------------------
def run_experiment(label_name, device):
    print(f"\n=== Executing experiment for: {label_name} ===")
    # Dá load no arquivo excel
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    raw = pd.read_excel(DATA_PATH)
    df, classes, target_names = prepare_dataframe_for_label(raw, label_name)

    # Checa equlíbrio de classes
    counts = df["label"].value_counts().sort_index()
    print("Class counts:", counts.to_dict())
    if df["label"].nunique() < 2:
        print("Not enough classes found for:", label_name)
        return

    # Divisão de treino e teste
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"])

    print(f"Sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Tokenizador + Datasets
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_ds = CommentDataset(train_df["comment_text"], train_df["label"], tokenizer, max_length=MAX_LENGTH)
    val_ds = CommentDataset(val_df["comment_text"], val_df["label"], tokenizer, max_length=MAX_LENGTH)
    test_ds = CommentDataset(test_df["comment_text"], test_df["label"], tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(classes)
    model = BertClassifier(MODEL_NAME, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} — train_loss: {tr_loss:.4f} train_acc: {tr_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

    # Salva as parcelas
    out_dir = os.path.join(OUTPUT_DIR, label_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='train')
    plt.plot(range(1, EPOCHS+1), val_losses, marker='s', label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss — {label_name}")
    plot_path = os.path.join(out_dir, f"loss_{label_name}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved loss plot to", plot_path)

    # Teste
    trues, preds = test_model(model, test_loader, device)

    # Relatório de classificação
    # Garante o mapeamento do ID numérico dos labels para target_names (target_names tem que alinhar!)
    # Para onca/caseiro usamos 0 = negativo, 1 = neutro, 2 = positivo
    print("\nClassification report:")
    print(classification_report(trues, preds, target_names=target_names, digits=4))

    # Salva o relatório em um arquivo txt
    report_txt = classification_report(trues, preds, target_names=target_names, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    # Salva o modelo and tokenizador
    model_save_path = os.path.join(out_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_save_path)
    tokenizer.save_pretrained(out_dir)
    print("Saved model and tokenizer to", out_dir)

    # Salva alguns erros em um arquivo txt
    test_texts = test_df["comment_text"].tolist()
    errors = []
    for t, y_true, y_pred in zip(test_texts, trues, preds):
        if y_true != y_pred:
            errors.append((t, target_names[y_true], target_names[y_pred]))
    errors_path = os.path.join(out_dir, "errors.txt")
    with open(errors_path, "w", encoding="utf-8") as f:
        for txt, t_true, t_pred in errors[:200]:
            f.write(f"TRUE: {t_true}\nPRED: {t_pred}\nTEXT: {txt}\n\n")
    print(f"Saved {min(len(errors),200)} error examples to {errors_path}")

# --------------------------
# O Main CLI
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, default="onca", choices=["onca","caseiro","fake_news","all"], help="label to train")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto by default)")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.label == "all":
        for lb in ["onca","caseiro","fake_news"]:
            run_experiment(lb, device)
    else:
        run_experiment(args.label, device)

if __name__ == "__main__":
    main()
