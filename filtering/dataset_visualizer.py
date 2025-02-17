import pandas as pd
import matplotlib.pyplot as plt


# Function to plot two CSV files
def plot_comparison(csv_file1, csv_file2):
    # Read the first CSV file
    df1 = pd.read_csv(csv_file1)
    # Read the second CSV file
    df2 = pd.read_csv(csv_file2)

    # Check if both dataframes have 3 columns
    if len(df1.columns) != 3 or len(df2.columns) != 3:
        raise ValueError("Both CSV files must have exactly 3 columns.")

    # Get the number of rows in the dataframes
    num_rows = min(len(df1), len(df2))

    # Create a figure with 3 subplots (one for each column)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot each column
    for i, col in enumerate(df1.columns):
        axes[i].plot(df1[col][:num_rows], label=f'{csv_file1} - {col}')
        axes[i].plot(df2[col][:num_rows], label=f'{csv_file2} - {col}')
        axes[i].set_title(f'Comparison of {col}')
        axes[i].set_xlabel('Index')
        axes[i].set_ylabel('Value')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace these with the actual paths to your CSV files
    csv_file1 = "dataset/CNNTransformerNet_Optuna_lr6.89e-05_bs16.csv"
    csv_file2 = "dataset/CNNTransformerNet_Optuna_lr6.89e-05_bs16.kalman.csv"

    plot_comparison(csv_file1, csv_file2)

