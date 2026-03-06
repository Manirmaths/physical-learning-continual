# src/make_figures.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load saved weights
    wA = np.load("w_taskA.npy")
    wAB_free = np.load("w_taskAB_free.npy")
    wAB_stable = np.load("w_taskAB_stable.npy")

    # 1) Parameter-change "learning signal"
    d_free = np.abs(wAB_free - wA)
    d_stable = np.abs(wAB_stable - wA)

    plt.figure()
    plt.title("Learning signal: |Δw| after Task B")
    x = np.arange(len(wA))
    plt.plot(x, d_free, label="B (free)")
    plt.plot(x, d_stable, label="B (stable)")
    plt.xlabel("Edge index")
    plt.ylabel("|Δw_raw|")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_learning_signal.png", dpi=200)

    # 2) Forgetting bars (use your printed numbers)
    mse_A_after_A = 8.6773e-07
    mse_A_after_B = 1.5787e-03
    mse_A_after_Bs = 1.0534e-03

    forgetting_free = mse_A_after_B - mse_A_after_A
    forgetting_stable = mse_A_after_Bs - mse_A_after_A

    plt.figure()
    plt.title("Forgetting on Task A after learning Task B")
    plt.bar(["B (free)", "B (stable)"], [forgetting_free, forgetting_stable])
    plt.ylabel("ΔMSE on Task A")
    plt.tight_layout()
    plt.savefig("fig_forgetting.png", dpi=200)

    print("Saved: fig_learning_signal.png, fig_forgetting.png")

if __name__ == "__main__":
    main()