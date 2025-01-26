
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=None):
    """Uses sequential splits for the dataset instead of random splits."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # Sequential splits
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    return (Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices))