"""
Training pipeline CLI and orchestrator for Ban Quality ML.
Usage:
    python -m ml.train [--data data/data_ban.xlsx] [--sheet data_ban] [--epochs 200] [--batch-size 32] [--lr 0.001] [--seed 42]
"""

import argparse
import random
import copy
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.config import (
    FEATURES,
    TARGET,
    DEFAULT_DATA_FILE,
    DEFAULT_DATA_SHEET,
    ARTIFACTS_DIR,
    MODEL_VERSION,
)
from ml.data import load_dataset, validate_dataset
from ml.preprocessing import split_dataset, fit_scaler, serialize_scaler
from ml.model import BanQualityMLP
from ml.evaluate import (
    compute_metrics,
    find_best_threshold_on_validation,
    perform_error_analysis,
    evaluate_baseline_logistic_regression,
)
from ml.artifacts import save_all_artifacts


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training_pipeline(
    data_path: str | Path = DEFAULT_DATA_FILE,
    sheet_name: str = DEFAULT_DATA_SHEET,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 20,
    seed: int = 42,
    threshold_mode: str = "val_f1",
    fixed_threshold: float = 0.50,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
) -> Dict[str, Any]:
    """
    Execute full reproducible training and evaluation pipeline.
    """
    set_seed(seed)
    print(f"[*] Starting Ban Quality ML Training (Seed: {seed})")

    # 1. Load and validate data
    raw_df = load_dataset(data_path, sheet_name=sheet_name)
    cleaned_df, audit_report = validate_dataset(raw_df, is_training=True)
    print(f"[+] Loaded {len(cleaned_df)} records from {data_path} [{sheet_name}].")
    print(f"    Label distribution: {audit_report['label_distribution']}")

    # 2. Stratified split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = split_dataset(cleaned_df, test_size=0.30, val_ratio=0.50, random_state=seed)
    print(f"[+] Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Fit scaler on train only
    scaler = fit_scaler(train_df)
    scaler_dict = serialize_scaler(scaler, FEATURES)

    # 4. Transform features
    X_train = scaler.transform(train_df[list(FEATURES)].to_numpy())
    y_train = train_df[TARGET].to_numpy().astype(np.float32)

    X_val = scaler.transform(val_df[list(FEATURES)].to_numpy())
    y_val = val_df[TARGET].to_numpy().astype(np.float32)

    X_test = scaler.transform(test_df[list(FEATURES)].to_numpy())
    y_test = test_df[TARGET].to_numpy().astype(np.float32)

    # Calculate pos_weight from train set: negative_count / positive_count
    pos_count = float(np.sum(y_train == 1.0))
    neg_count = float(np.sum(y_train == 0.0))
    pos_weight_val = (neg_count / pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"[+] Calculated train pos_weight: {pos_weight_val:.3f} (neg={neg_count:.0f}, pos={pos_count:.0f})")

    # PyTorch Datasets & DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Build Model, Criterion, Optimizer
    model = BanQualityMLP(input_dim=len(FEATURES), hidden_layers=[32, 16], dropout=0.10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0
    training_history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.50).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_x)

        epoch_train_loss = train_loss_total / train_total
        epoch_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss_total += loss.item() * len(batch_x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.50).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_x)

        epoch_val_loss = val_loss_total / val_total
        epoch_val_acc = val_correct / val_total

        history_entry = {
            "epoch": epoch,
            "train_loss": round(float(epoch_train_loss), 4),
            "val_loss": round(float(epoch_val_loss), 4),
            "train_accuracy": round(float(epoch_train_acc), 4),
            "val_accuracy": round(float(epoch_val_acc), 4),
        }
        training_history.append(history_entry)

        # Checkpoint evaluation
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[*] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, val_loss: {best_val_loss:.4f})")
            break

    # Restore best checkpoint
    model.load_state_dict(best_model_weights)
    model.eval()

    # 7. Select threshold on validation set
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        val_logits = model(val_tensor)
        val_probs = torch.sigmoid(val_logits).squeeze().numpy()

    if threshold_mode == "val_f1":
        selected_threshold = find_best_threshold_on_validation(y_val, val_probs)
    elif threshold_mode == "fixed":
        selected_threshold = fixed_threshold
    else:
        selected_threshold = 0.50

    print(f"[+] Selected decision threshold on validation set: {selected_threshold:.2f}")

    # 8. Single final evaluation on holdout test set
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        test_logits = model(test_tensor)
        test_probs = torch.sigmoid(test_logits).squeeze().numpy()

    test_metrics = compute_metrics(y_test, test_probs, threshold=selected_threshold)
    y_test_pred = (test_probs >= selected_threshold).astype(int)
    error_analysis = perform_error_analysis(test_df, y_test, y_test_pred, test_probs)

    # Baseline comparison (Logistic Regression)
    baseline_eval = evaluate_baseline_logistic_regression(X_train, y_train, X_test, y_test)

    # 9. Assemble evaluation report
    evaluation_report = {
        "model_version": MODEL_VERSION,
        "dataset": {
            "file": Path(data_path).name,
            "sheet": sheet_name,
            "rows": len(cleaned_df),
        },
        "features": list(FEATURES),
        "target": TARGET,
        "split": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "random_state": seed,
        },
        "training_summary": {
            "best_epoch": best_epoch,
            "total_epochs_trained": len(training_history),
            "best_val_loss": round(float(best_val_loss), 4),
            "pos_weight": round(pos_weight_val, 4),
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
        },
        "threshold": selected_threshold,
        "metrics": test_metrics,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "baseline_comparison": baseline_eval,
        "error_analysis": error_analysis,
    }

    # 10. Save all artifacts
    artifact_paths = save_all_artifacts(
        model=model,
        scaler_dict=scaler_dict,
        eval_dict=evaluation_report,
        history_list=training_history,
        threshold=selected_threshold,
        artifacts_dir=artifacts_dir,
        model_version=MODEL_VERSION,
    )

    print("\n" + "=" * 50)
    print("           TRAINING & EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"Precision (Class 1): {test_metrics['precision_class_1']:.4f}")
    print(f"Recall (Class 1):    {test_metrics['recall_class_1']:.4f}")
    print(f"F1-Score (Class 1):  {test_metrics['f1_class_1']:.4f}")
    if test_metrics['roc_auc'] is not None:
        print(f"ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    print(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"Confusion Matrix:   {test_metrics['confusion_matrix']}")
    print("=" * 50)
    print(f"[+] Artifacts saved to: {artifacts_dir}")
    for k, v in artifact_paths.items():
        print(f"    - {k}: {v.name}")

    return evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Train Ban Quality MLP Classifier")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_FILE), help="Path to data_ban.xlsx")
    parser.add_argument("--sheet", type=str, default=DEFAULT_DATA_SHEET, help="Sheet name")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold-mode", type=str, choices=["val_f1", "fixed"], default="val_f1", help="Threshold selection method")
    parser.add_argument("--threshold", type=float, default=0.50, help="Fixed threshold value if mode is fixed")
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACTS_DIR), help="Output artifacts directory")

    args = parser.parse_args()
    run_training_pipeline(
        data_path=args.data,
        sheet_name=args.sheet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        threshold_mode=args.threshold_mode,
        fixed_threshold=args.threshold,
        artifacts_dir=args.artifacts,
    )


if __name__ == "__main__":
    main()
