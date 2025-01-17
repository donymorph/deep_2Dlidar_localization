
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Splits a dataset into train, val, test subsets according to given ratios.
    Default: 70% train, 20% val, 10% test.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    total_size = len(dataset)
    indices = list(range(total_size))

    # First split -> train + remainder
    train_size = int(train_ratio * total_size)
    remainder_size = total_size - train_size

    train_indices, remainder_indices = train_test_split(
        indices, 
        train_size=train_size, 
        random_state=random_seed
    )

    # Second split -> val + test
    val_size = int(val_ratio / (val_ratio + test_ratio) * remainder_size)
    val_indices, test_indices = train_test_split(
        remainder_indices, 
        train_size=val_size, 
        random_state=random_seed
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset
