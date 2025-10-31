import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from models.regime_transformer_baseline import RegimeTransformer

  # adjust if path different

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    """
    Adjust this to how your teammate saved the data.
    For now I am assuming .pt files in data/processed/
    """
    data_dir = Path("../data") / "processed"

    X_train = torch.load(data_dir / "X_train.pt")  # (N, seq_len, input_dim)
    y_train = torch.load(data_dir / "y_train.pt")  # (N,)
    X_val = torch.load(data_dir / "X_val.pt")
    y_val = torch.load(data_dir / "y_val.pt")
    return X_train, y_train, X_val, y_val

def main():
    X_train, y_train, X_val, y_val = load_data()

    # our notebook saved (N, 9); transformer wants (N, T, F)
    if X_train.ndim == 2:
        X_train = X_train.unsqueeze(1)   # (N, 1, F)
        X_val = X_val.unsqueeze(1)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)


    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    input_dim = X_train.shape[-1]
    num_classes = int(y_train.max().item()) + 1  # expects labels 0..2

    model = RegimeTransformer(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_true.append(yb.cpu())

        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)
        val_acc = accuracy_score(all_true, all_preds)
        val_f1 = f1_score(all_true, all_preds, average="macro")

        print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_dir = Path("artifacts")
            out_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), out_dir / "regime_transformer_baseline.pt")

if __name__ == "__main__":
    main()
