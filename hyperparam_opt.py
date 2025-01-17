import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

from training import train_model

# Optional: For Bayesian optimization with scikit-optimize
try:
    from skopt import gp_minimize
    from skopt.space import Real, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not installed. Bayesian optimization won't run.")



#########################################################
# 1) Grid Search
#########################################################
def grid_search_hyperparams():
    model_choices = ['ConvTransformerNet']  # Example: ['SimpleMLP', 'DeeperMLP', 'Conv1DNet']
    lrs = [1e-4, 1e-3, 1e-2]
    batch_sizes = [16, 32, 64]

    best_loss = float('inf')
    best_config = None

    results = []  # Store (config, final_loss, epoch_times)
    for model_choice in model_choices:
        for lr in lrs:
            for batch_size in batch_sizes:
                print(f"\n>>> [Grid] model={model_choice}, lr={lr}, batch_size={batch_size} <<<")
                final_test_loss, epoch_times = train_model(
                    odom_csv="odom_data.csv",
                    scan_csv="scan_data.csv",
                    model_choice=model_choice,
                    batch_size=batch_size,
                    lr=lr,
                    epochs=1,
                    train_ratio=0.7,
                    val_ratio=0.2,
                    test_ratio=0.1,
                    random_seed=42,
                    log_dir=f"tensorboard_logs/grid_{model_choice}_lr{lr}_bs{batch_size}"
                )
                results.append(((model_choice, lr, batch_size), final_test_loss, epoch_times))
                if final_test_loss < best_loss:
                    best_loss = final_test_loss
                    best_config = (model_choice, lr, batch_size)

    print("\n=== Grid Search Complete ===")
    print(f"Best Config: model={best_config[0]}, lr={best_config[1]}, batch_size={best_config[2]}")
    print(f"Best Final Test Loss: {best_loss:.6f}")
    return results

#########################################################
# 2) Random Search
#########################################################
def random_search_hyperparams(num_experiments=5):
    model_choices = ['ConvTransformerNet']
    lr_range = (1e-4, 1e-2)
    batch_size_options = [16, 32, 64]

    best_loss = float('inf')
    best_config = None
    results = []

    for i in range(num_experiments):
        model_choice = random.choice(model_choices)
        lr = 10 ** np.random.uniform(np.log10(lr_range[0]), np.log10(lr_range[1]))
        batch_size = random.choice(batch_size_options)

        print(f"\n>>> [Random] Iter={i+1}/{num_experiments}, model={model_choice}, lr={lr:.6g}, batch_size={batch_size} <<<")
        final_test_loss, epoch_times = train_model(
            odom_csv="odom_data.csv",
            scan_csv="scan_data.csv",
            model_choice=model_choice,
            batch_size=batch_size,
            lr=lr,
            epochs=5,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
            log_dir=f"tensorboard_logs/random_{model_choice}_iter{i+1}"
        )
        results.append(((model_choice, lr, batch_size), final_test_loss, epoch_times))
        if final_test_loss < best_loss:
            best_loss = final_test_loss
            best_config = (model_choice, lr, batch_size)

    print("\n=== Random Search Complete ===")
    print(f"Best Config: model={best_config[0]}, lr={best_config[1]}, batch_size={best_config[2]}")
    print(f"Best Final Test Loss: {best_loss:.6f}")
    return results

#########################################################
# 3) Bayesian Optimization (via scikit-optimize)
#########################################################
def bayesian_opt_hyperparams(n_calls=10):
    """
    Example usage of scikit-optimize's gp_minimize.
    """
    if not SKOPT_AVAILABLE:
        print("scikit-optimize not installed, skipping Bayesian optimization.")
        return []


    # Define the search space
    space = [
        Categorical(['ConvTransformerNet'], name='model_choice'),
        Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
        Categorical([16, 32, 64], name='batch_size'),
    ]

    @use_named_args(space)
    def objective(model_choice, lr, batch_size):
        print(f"\n>>> [Bayes] model={model_choice}, lr={lr:.6g}, batch_size={batch_size} <<<")
        final_test_loss, _ = train_model(
            odom_csv="odom_data.csv",
            scan_csv="scan_data.csv",
            model_choice=model_choice,
            batch_size=batch_size,
            lr=lr,
            epochs=5,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
            log_dir=f"tensorboard_logs/bayesian_{model_choice}_lr{lr}_bs{batch_size}"
        )
        return final_test_loss

    # Run gp_minimize
    res = gp_minimize(
        objective, 
        dimensions=space,
        n_calls=n_calls,
        random_state=42
    )

    print("\n=== Bayesian Opt Complete ===")
    print("Best hyperparameters:", res.x)
    print("Best loss:", res.fun)
    return res

#########################################################
# Plotting Results
#########################################################
def plot_results(search_results, title="Hyperparam Search"):
    """
    search_results: list of ( (model_choice, lr, batch_size), final_loss, epoch_times )
    """
    iters = range(1, len(search_results)+1)
    losses = [r[1] for r in search_results]

    plt.figure(figsize=(6, 4))
    plt.plot(iters, losses, 'o-', label='Final Loss')
    plt.title(title)
    plt.xlabel('Search Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

#########################################################
# Main
#########################################################
def main():
    # 1) Grid Search
    grid_results = grid_search_hyperparams()
    #plot_results(grid_results, title="Grid Search Results")

    # 2) Random Search
    random_results = random_search_hyperparams(num_experiments=5)
    #plot_results(random_results, title="Random Search Results")

    # 3) Bayesian Optimization
    if SKOPT_AVAILABLE:
        bayesian_results = bayesian_opt_hyperparams(n_calls=10)

if __name__ == "__main__":
    main()
