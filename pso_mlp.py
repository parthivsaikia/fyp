"""
PSO-based hyperparameter optimization for MLP classifier.
"""
import numpy as np
import pickle
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pyswarms as ps
from joblib import dump, load

from config import (
    PSO_CONFIG, HIDDEN_CONFIGS, ACTIVATIONS,
    CV_FOLDS, CV_SCORING, MLP_MAX_ITER, MLP_EARLY_STOPPING,
    MLP_VALIDATION_FRACTION, MLP_N_ITER_NO_CHANGE, MLP_RANDOM_STATE,
    OUTPUT_DIR, RANDOM_STATE
)


def decode_particle(position: np.ndarray) -> Dict[str, Any]:
    """Decode PSO particle position to MLP hyperparameters."""
    # Position: [hidden_config_idx, alpha, learning_rate_init, activation_idx]
    hidden_idx = int(np.clip(round(position[0]), 0, len(HIDDEN_CONFIGS) - 1))
    alpha = float(np.clip(position[1], PSO_CONFIG["bounds"]["alpha"][0], PSO_CONFIG["bounds"]["alpha"][1]))
    lr = float(np.clip(position[2], PSO_CONFIG["bounds"]["learning_rate_init"][0], PSO_CONFIG["bounds"]["learning_rate_init"][1]))
    act_idx = int(np.clip(round(position[3]), 0, len(ACTIVATIONS) - 1))
    
    return {
        "hidden_layer_sizes": HIDDEN_CONFIGS[hidden_idx],
        "alpha": alpha,
        "learning_rate_init": lr,
        "activation": ACTIVATIONS[act_idx],
    }


def create_mlp(params: Dict[str, Any], random_state: int = MLP_RANDOM_STATE) -> MLPClassifier:
    """Create MLP classifier with given hyperparameters."""
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        activation=params["activation"],
        max_iter=MLP_MAX_ITER,
        early_stopping=MLP_EARLY_STOPPING,
        validation_fraction=MLP_VALIDATION_FRACTION,
        n_iter_no_change=MLP_N_ITER_NO_CHANGE,
        random_state=random_state,
        solver="adam",
        batch_size="auto",
        learning_rate="adaptive",
    )


def fitness_function(positions: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    PSO fitness function - negative CV score (PSO minimizes).
    
    Args:
        positions: (n_particles, n_dimensions)
        X: Feature matrix
        y: Labels
        
    Returns:
        Fitness values (negative CV score for minimization)
    """
    n_particles = positions.shape[0]
    fitness = np.zeros(n_particles)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for i in range(n_particles):
        params = decode_particle(positions[i])
        mlp = create_mlp(params)
        
        # Create pipeline with scaler
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", mlp)
        ])
        
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1)
            fitness[i] = -np.mean(scores)  # Negative for minimization
        except Exception as e:
            print(f"  Particle {i} failed: {e}")
            fitness[i] = 1.0  # Worst fitness
    
    return fitness


def optimize_pso(
    X: np.ndarray,
    y: np.ndarray,
    sensor_name: str,
    n_particles: int = None,
    n_iterations: int = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[float], np.ndarray]:
    """
    Run PSO optimization to find best MLP hyperparameters.
    
    Returns:
        best_params: Best hyperparameters found
        convergence_curve: Best fitness per iteration
        best_position: Best particle position
    """
    n_particles = n_particles or PSO_CONFIG["n_particles"]
    n_iterations = n_iterations or PSO_CONFIG["n_iterations"]
    
    # Bounds for PSO
    bounds = (
        np.array([b[0] for b in PSO_CONFIG["bounds"].values()]),
        np.array([b[1] for b in PSO_CONFIG["bounds"].values()])
    )
    
    # PSO options
    options = {
        "c1": PSO_CONFIG["c1"],
        "c2": PSO_CONFIG["c2"],
        "w": PSO_CONFIG["w"],
    }
    
    # Create optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=4,
        options=options,
        bounds=bounds,
        ftol=1e-6,
        ftol_iter=10,
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PSO Optimization for {sensor_name}")
        print(f"Particles: {n_particles}, Iterations: {n_iterations}")
        print(f"CV Folds: {CV_FOLDS}, Scoring: {CV_SCORING}")
        print(f"{'='*60}\n")
    
    # Run optimization
    start_time = time.time()
    
    def obj_func(pos):
        return fitness_function(pos, X, y)
    
    best_cost, best_pos = optimizer.optimize(obj_func, iters=n_iterations, verbose=verbose)
    
    elapsed = time.time() - start_time
    
    # Decode best position
    best_params = decode_particle(best_pos)
    best_cv_score = -best_cost
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PSO completed in {elapsed:.1f}s")
        print(f"Best CV {CV_SCORING}: {best_cv_score:.4f}")
        print(f"Best params: {best_params}")
        print(f"{'='*60}\n")
    
    return best_params, optimizer.cost_history, best_pos


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict[str, Any],
    sensor_name: str,
    save_path: Optional[Path] = None
) -> Tuple[Pipeline, StandardScaler]:
    """
    Train final MLP model on full training set with best hyperparameters.
    
    Returns:
        pipeline: Fitted sklearn Pipeline (scaler + MLP)
        scaler: Fitted StandardScaler
    """
    mlp = create_mlp(best_params)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp)
    ])
    
    print(f"\nTraining final model for {sensor_name}...")
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
    pipeline.fit(X_train, y_train)
    
    # Extract scaler and MLP for saving
    scaler = pipeline.named_steps["scaler"]
    trained_mlp = pipeline.named_steps["mlp"]
    
    print(f"Training completed.")
    print(f"  Iterations: {trained_mlp.n_iter_}")
    print(f"  Final loss: {trained_mlp.loss_:.4f}")
    print(f"  Best validation score: {trained_mlp.best_validation_score_:.4f}" if hasattr(trained_mlp, 'best_validation_score_') else "")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        dump(pipeline, save_path / f"{sensor_name}_pipeline.joblib")
        
        # Save scaler separately
        dump(scaler, save_path / f"{sensor_name}_scaler.joblib")
        
        # Save MLP separately
        dump(trained_mlp, save_path / f"{sensor_name}_mlp.joblib")
        
        # Save hyperparameters
        with open(save_path / f"{sensor_name}_best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        print(f"Saved artifacts to {save_path}")
    
    return pipeline, scaler


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_name: str,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """Evaluate model on test set."""
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        accuracy_score, f1_score, precision_score, recall_score
    )
    
    y_pred = pipeline.predict(X_test)
    
    if class_names is None:
        class_names = [str(i) for i in np.unique(y_test)]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    results = {
        "sensor": sensor_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_samples": int(len(y_test)),
    }
    
    print(f"\n{'='*60}")
    print(f"Test Evaluation - {sensor_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Macro Prec:{macro_precision:.4f}")
    print(f"Macro Rec: {macro_recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    return results


def save_results(results: Dict, sensor_name: str, output_dir: Path):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{sensor_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for easy reading
    import pandas as pd
    report = results["classification_report"]
    df = pd.DataFrame(report).T
    df.to_csv(output_dir / f"{sensor_name}_classification_report.csv")
    
    print(f"Results saved to {output_dir}")


def get_training_loss_curve(pipeline: Pipeline) -> List[float]:
    """Get training loss curve from fitted MLP."""
    mlp = pipeline.named_steps["mlp"]
    if hasattr(mlp, "loss_curve_"):
        loss_curve = mlp.loss_curve_
        if hasattr(loss_curve, 'tolist'):
            return loss_curve.tolist()
        return list(loss_curve)
    return []


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=20, n_classes=3, random_state=42)
    
    best_params, curve, best_pos = optimize_pso(X, y, "test", n_particles=5, n_iterations=3)
    print(f"Best params: {best_params}")
    print(f"Convergence: {curve}")