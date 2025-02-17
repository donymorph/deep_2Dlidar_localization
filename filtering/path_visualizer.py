import pandas as pd
import matplotlib.pyplot as plt


def plot_three_csv(csv_file1, csv_file2, csv_file3, max_points:int=200):
    try:
        # Read the first CSV file
        df1 = pd.read_csv(csv_file1)
        # Read the second CSV file
        df2 = pd.read_csv(csv_file2)
        # Read the third CSV file
        df3 = pd.read_csv(csv_file3)

        # Check if each dataframe has the required columns
        for df in [df1, df2, df3]:
            if "pos_x" not in df.columns or "pos_y" not in df.columns:
                raise ValueError("Each CSV file must have 'pos_x' and 'pos_y' columns.")

        # Extract x and y values from each dataframe

        x1, y1 = df1["pos_x"][:max_points], df1["pos_y"][:max_points]
        x2, y2 = df2["pos_x"][:max_points], df2["pos_y"][:max_points]
        x3, y3 = df3["pos_x"][:max_points], df3["pos_y"][:max_points]

        # Create a new figure
        plt.figure(figsize=(10, 6))

        # Plot the data from each CSV file
        plt.plot(x1, y1, label=csv_file1, marker='o', linestyle='-')
        plt.plot(x2, y2, label=csv_file2, marker='s', linestyle='--')
        plt.plot(x3, y3, label=csv_file3, marker='^', linestyle='-.')

        # Add title and labels
        plt.title('Comparison of Three CSV Files')
        plt.xlabel('pos_x values')
        plt.ylabel('pos_y values')

        # Add legend
        plt.legend()

        # Show the grid
        plt.grid(True)

        # Display the plot
        plt.show()
    except FileNotFoundError:
        print("One or more of the specified CSV files were not found.")
    except ValueError as ve:
        print(ve)


if __name__ == "__main__":
    # Replace these with the actual paths to your CSV files
    csv_file1 = "dataset/processed_odom.csv"
    csv_file2 = "dataset/CNNTransformerNet_Optuna_lr6.89e-05_bs16.csv"
    csv_file3 = "dataset/CNNTransformerNet_Optuna_lr6.89e-05_bs16.kalman.csv"

    plot_three_csv(csv_file1, csv_file2, csv_file3, max_points=300)
