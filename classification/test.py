# evaluate_separate_test.py
import os
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from train import CNN1DModel  

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PyTorch 1D‑CNN on hold‑out test set"
    )
    parser.add_argument('--models_dir', type=str, default='Models')
    parser.add_argument('--n_splits',   type=int, default=4)
    parser.add_argument('--fruit',      type=str, default='milk')
    args = parser.parse_args()
    
    # load test split
    X_test = np.load(os.path.join(args.models_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.models_dir, "y_test.npy"))
    print(f"Test set: {X_test.shape[0]} samples\n")
    
    fold_accuracies = []
    fold_precisions = []
    fold_recalls    = []
    
    for fold in range(args.n_splits):
        ckpt_path = os.path.join(args.models_dir, f"CNN1D_{args.fruit}_fold{fold}.pt")
        print(f"--- Evaluating fold {fold} ---")
        
        # load model + scaler
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        scaler = state['scaler']
        
        # rebuild network with the same number of classes as in checkpoint
        saved_fc2   = state['model_state_dict']['fc2.weight']
        num_classes = saved_fc2.shape[0]
        net = CNN1DModel(input_channels=1, num_classes=num_classes)
        net.load_state_dict(state['model_state_dict'])
        net.eval()
        
        # scale and convert to tensor
        Xs = scaler.transform(X_test)
        Xs = torch.tensor(Xs, dtype=torch.float32)
        
        # forward pass
        with torch.no_grad():
            logits = net(Xs)
            y_pred = logits.argmax(dim=1).numpy()
        
        # compute metrics
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        cm        = confusion_matrix(y_test, y_pred)
        rep       = classification_report(y_test, y_pred, digits=4)
        
        fold_accuracies.append(acc)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        
        print(f"Accuracy:           {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(rep)
        print()
    
    # aggregate
    mean_acc  = np.mean(fold_accuracies)
    std_acc   = np.std(fold_accuracies)
    mean_prec = np.mean(fold_precisions)
    std_prec  = np.std(fold_precisions)
    mean_rec  = np.mean(fold_recalls)
    std_rec   = np.std(fold_recalls)
    
    print("="*60)
    print("ACADEMIC PAPER METRICS - AGGREGATED RESULTS")
    print("="*60)
    print(f"Model Performance on {args.fruit.upper()} dataset:")
    print(f"  Folds: {args.n_splits}, Test samples: {X_test.shape[0]}, Classes: {num_classes}\n")
    print("Cross‑validation results (mean ± std):")
    print(f"  Accuracy:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Precision: {mean_prec*100:.2f}% ± {std_prec*100:.2f}%")
    print(f"  Recall:    {mean_rec*100:.2f}% ± {std_rec*100:.2f}%\n")
    print("Individual fold results:")
    for i, (a,p,r) in enumerate(zip(fold_accuracies, fold_precisions, fold_recalls)):
        print(f"  Fold {i}: Acc={a*100:.2f}%, Prec={p*100:.2f}%, Rec={r*100:.2f}%")
    print()
    print("For academic paper:")
    print(f"\"The CNN1D achieved an average accuracy of {mean_acc*100:.0f}% ± {std_acc*100:.1f}%, "
          f"precision of {mean_prec*100:.0f}% ± {std_prec*100:.1f}%, and recall of {mean_rec*100:.0f}% ± {std_rec*100:.1f}% "
          f"on the {args.fruit.upper()} test dataset using {args.n_splits}-fold cross-validation.\"")

if __name__ == "__main__":
    main()
