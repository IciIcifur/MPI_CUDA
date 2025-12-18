import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def plot_trajectories(csv_file, show_plot=False):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    n_columns = len(df.columns)
    n_bodies = (n_columns - 1) // 2

    print(f"Loaded {len(df)} time steps")
    print(f"Detected {n_bodies} bodies")

    csv_path = Path(csv_file)
    output_dir = csv_path.parent

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Trajectories of {n_bodies} bodies (N-body simulation)',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))

    ax1 = axes[0, 0]

    for i in range(n_bodies):
        x_col = f'x{i+1}'
        y_col = f'y{i+1}'
        ax1.plot(df[x_col], df[y_col],
                 color=colors[i],
                 linewidth=1.5,
                 alpha=0.7,
                 label=f'Body {i+1}')

    ax1.set_xlabel('X coordinate (m)')
    ax1.set_ylabel('Y coordinate (m)')
    ax1.set_title('2D trajectories of all bodies')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.axis('equal')

    ax2 = axes[0, 1]

    for i in range(n_bodies):
        x_col = f'x{i+1}'
        y_col = f'y{i+1}'
        distance = np.sqrt(df[x_col]**2 + df[y_col]**2)
        ax2.plot(df['t'], distance,
                 color=colors[i],
                 linewidth=1.5,
                 alpha=0.7,
                 label=f'Body {i+1}')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance from origin (m)')
    ax2.set_title('Distance from (0,0) vs time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    ax3 = axes[1, 0]
    for i in range(n_bodies):
        x_col = f'x{i+1}'
        ax3.plot(df['t'], df[x_col],
                 color=colors[i],
                 linewidth=1.5,
                 alpha=0.7,
                 label=f'Body {i+1}')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('X coordinate (m)')
    ax3.set_title('X coordinates vs time')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    ax4 = axes[1, 1]
    for i in range(n_bodies):
        y_col = f'y{i+1}'
        ax4.plot(df['t'], df[y_col],
                 color=colors[i],
                 linewidth=1.5,
                 alpha=0.7,
                 label=f'Body {i+1}')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Y coordinate (m)')
    ax4.set_title('Y coordinates vs time')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')

    plt.tight_layout()

    output_path = output_dir / 'trajectories_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    report_path = output_dir / 'simulation_report.txt'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== N-BODY SIMULATION REPORT ===\n\n")
            f.write(f"Data file: {csv_file}\n")
            f.write(f"Number of bodies: {n_bodies}\n")
            f.write(f"Number of time steps: {len(df)}\n")
            f.write(f"Simulation time: from {df['t'].iloc[0]} to {df['t'].iloc[-1]} s\n")

            if len(df) > 1:
                dt = df['t'].iloc[1] - df['t'].iloc[0]
                f.write(f"Time step: {dt:.2f} s\n")

            f.write("\nFINAL POSITIONS:\n")
            for i in range(n_bodies):
                x_col = f'x{i+1}'
                y_col = f'y{i+1}'
                x_init = df[x_col].iloc[0]
                y_init = df[y_col].iloc[0]
                x_final = df[x_col].iloc[-1]
                y_final = df[y_col].iloc[-1]

                f.write(f"Body {i+1}:\n")
                f.write(f"  Initial: ({x_init:.2e}, {y_init:.2e}) m\n")
                f.write(f"  Final:   ({x_final:.2e}, {y_final:.2e}) m\n")

                displacement = np.sqrt((x_final - x_init)**2 + (y_final - y_init)**2)
                f.write(f"  Displacement: {displacement:.2e} m\n\n")

        print(f"Report saved: {report_path}")

    except Exception as e:
        print(f"Warning: Could not create report file: {e}")

    return True

def main():
    parser = argparse.ArgumentParser(description='Visualize N-body trajectories')
    parser.add_argument('--file', '-f', required=True,
                        help='CSV file with trajectories (e.g., trajectories.csv)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots on screen')
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help='Do not show plots on screen (default)')
    parser.set_defaults(show=False)

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        return 1

    try:
        success = plot_trajectories(args.file, args.show)
        if success:
            print("Visualization completed successfully!")
            return 0
        else:
            return 1
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())