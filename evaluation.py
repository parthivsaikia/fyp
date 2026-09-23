"""
Evaluation and visualization module for MLP fault diagnosis.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.inspection import permutation_importance

from config import FAULT_CLASSES, OUTPUT_DIR, TOP_K_FEATURES, N_PERMUTATION_REPEATS, RANDOM_STATE


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    sensor_name: str,
    save_path: Path,
    normalize: bool = True
):
    """Plot and save confusion matrix heatmap."""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        title = f'{sensor_name} - Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = f'{sensor_name} - Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_pso_convergence(
    convergence_curve: List[float],
    sensor_name: str,
    save_path: Path
):
    """Plot PSO convergence curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(convergence_curve, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (Negative CV Score)', fontsize=12)
    plt.title(f'{sensor_name} - PSO Convergence', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmin(convergence_curve)
    plt.plot(best_idx, convergence_curve[best_idx], 'ro', markersize=10, 
             label=f'Best: {convergence_curve[best_idx]:.4f} at iter {best_idx}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PSO convergence to {save_path}")


def plot_training_loss(
    loss_curve: List[float],
    sensor_name: str,
    save_path: Path
):
    """Plot MLP training loss curve."""
    if not loss_curve:
        print("No loss curve available")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_curve, 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{sensor_name} - MLP Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training loss to {save_path}")


def plot_feature_importance(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    sensor_name: str,
    save_path: Path,
    top_k: int = TOP_K_FEATURES
):
    """Compute and plot permutation feature importance."""
    print(f"Computing permutation importance for {sensor_name}...")
    
    try:
        result = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring='f1_macro'
        )
        
        importances = result.importances_mean
        std = result.importances_std
        
        # Get top K
        top_indices = np.argsort(importances)[-top_k:][::-1]
        top_importances = importances[top_indices]
        top_std = std[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Plot
        plt.figure(figsize=(10, max(6, top_k * 0.3)))
        y_pos = np.arange(len(top_names))
        plt.barh(y_pos, top_importances, xerr=top_std, align='center', 
                 capsize=3, color='steelblue', edgecolor='black', alpha=0.7)
        plt.yticks(y_pos, top_names)
        plt.xlabel('Permutation Importance (Macro F1 decrease)', fontsize=12)
        plt.title(f'{sensor_name} - Top {top_k} Feature Importance', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save importance values
        importance_data = {
            "sensor": sensor_name,
            "top_features": [
                {"name": name, "importance": float(imp), "std": float(s)}
                for name, imp, s in zip(top_names, top_importances, top_std)
            ]
        }
        
        importance_path = save_path.with_suffix('.json')
        with open(importance_path, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        print(f"Saved feature importance to {save_path}")
        
    except Exception as e:
        print(f"Failed to compute feature importance: {e}")


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    sensor_name: str,
    save_path: Path,
    class_names: List[str] = None
):
    """Plot class distribution in train/test sets."""
    if class_names is None:
        class_names = FAULT_CLASSES
    
    train_counts = np.bincount(y_train, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Fault Class', fontsize=12)
    ax.set_ylabel('Number of Windows', fontsize=12)
    ax.set_title(f'{sensor_name} - Class Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution to {save_path}")


def save_feature_list(feature_names: List[str], sensor_name: str, save_path: Path):
    """Save feature names to file."""
    with open(save_path, 'w') as f:
        for i, name in enumerate(feature_names):
            f.write(f"{i}: {name}\n")
    print(f"Saved feature list to {save_path}")


def create_summary_report(
    results: Dict,
    sensor_name: str,
    output_dir: Path,
    class_names: List[str] = None
):
    """Create a text summary report."""
    if class_names is None:
        class_names = FAULT_CLASSES
    
    report_path = output_dir / f"{sensor_name}_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"MLP Fault Diagnosis - {sensor_name.upper()} Sensor\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Test Set Performance:\n")
        f.write(f"  Accuracy:     {results['accuracy']:.4f}\n")
        f.write(f"  Macro F1:     {results['macro_f1']:.4f}\n")
        f.write(f"  Macro Precision: {results['macro_precision']:.4f}\n")
        f.write(f"  Macro Recall:   {results['macro_recall']:.4f}\n\n")
        
        f.write(f"Test Samples: {results['test_samples']}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        report = results['classification_report']
        for cls in class_names:
            if cls in report:
                m = report[cls]
                f.write(f"  {cls:12s}: Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1-score']:.4f}, Support={m['support']}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        cm = np.array(results['confusion_matrix'])
        f.write(f"  Rows=Actual, Cols=Predicted\n")
        f.write(f"  Classes: {class_names}\n")
        for i, row in enumerate(cm):
            f.write(f"  {class_names[i]:12s}: {row}\n")
    
    print(f"Saved summary report to {report_path}")


def run_full_evaluation(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    sensor_name: str,
    convergence_curve: List[float],
    loss_curve: List[float],
    y_train: np.ndarray = None,
    output_dir: Path = None,
    class_names: List[str] = None
) -> Dict:
    """Run complete evaluation and generate all plots."""
    if class_names is None:
        class_names = FAULT_CLASSES
    
    if output_dir is None:
        output_dir = OUTPUT_DIR / sensor_name
    else:
        output_dir = Path(output_dir) / sensor_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    from pso_mlp import evaluate_model
    results = evaluate_model(pipeline, X_test, y_test, sensor_name, class_names)
    
    # Save results
    from pso_mlp import save_results
    save_results(results, sensor_name, output_dir)
    
    # Generate plots
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        class_names,
        sensor_name,
        output_dir / f"{sensor_name}_confusion_matrix.png"
    )
    
    plot_pso_convergence(
        convergence_curve,
        sensor_name,
        output_dir / f"{sensor_name}_pso_convergence.png"
    )
    
    plot_training_loss(
        loss_curve,
        sensor_name,
        output_dir / f"{sensor_name}_training_loss.png"
    )
    
    plot_feature_importance(
        pipeline,
        X_test,
        y_test,
        feature_names,
        sensor_name,
        output_dir / f"{sensor_name}_feature_importance.png"
    )
    
    if y_train is not None:
        plot_class_distribution(
            y_train,
            y_test,
            sensor_name,
            output_dir / f"{sensor_name}_class_distribution.png",
            class_names=class_names
        )
    
    save_feature_list(feature_names, sensor_name, output_dir / f"{sensor_name}_feature_list.txt")
    
    create_summary_report(results, sensor_name, output_dir, class_names=class_names)
    
    return results


if __name__ == "__main__":
    # Quick test
    import numpy as np
    cm = np.array([[50, 2, 1], [3, 45, 5], [2, 1, 48]])
    plot_confusion_matrix(cm, ['A', 'B', 'C'], 'test', Path('/tmp/test_cm.png'))
    print("Test plot saved")