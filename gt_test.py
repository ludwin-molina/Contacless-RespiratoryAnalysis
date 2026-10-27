import pandas as pd
import matplotlib.pyplot as plt

def plot_signals():
    file_path = r"C:\Users\domin\Desktop\pomiary_d\gt\processed_data-paulina-lidar.csv"
    
    try:
        df = pd.read_csv(file_path, sep=',')
    except FileNotFoundError:
        print(f"Błąd krytyczny: Plik docelowy nie istnieje pod ścieżką {file_path}")
        return

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(df['Time'], df['Raw'], color='gray', linewidth=1)
    axs[0].set_title('Raw Signal')
    axs[0].set_ylabel('Resistance [kOhm]')
    axs[0].grid(True)

    axs[1].plot(df['Time'], df['Filtered'], color='blue', linewidth=1)
    axs[1].set_title('Filtered Signal (0.05 - 1.5 Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    axs[2].plot(df['Time'], df['BPM'], color='red', linewidth=1.5)
    axs[2].set_title('Estimated BPM')
    axs[2].set_ylabel('BPM')
    axs[2].set_xlabel('Elapsed Time [s]')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_signals()