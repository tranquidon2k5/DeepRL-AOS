# scripts/plot_metrics.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def _stack_obj_series(obj_arr, num_ops: int):
    """
    obj_arr: numpy array(dtype=object) gồm nhiều vector dài num_ops
    Trả về matrix shape [T, num_ops], NaN nếu phần tử thiếu.
    """
    T = len(obj_arr)
    out = np.full((T, num_ops), np.nan, dtype=float)
    for t in range(T):
        v = obj_arr[t]
        if v is None:
            continue
        v = np.asarray(v, dtype=float).reshape(-1)
        k = min(num_ops, v.size)
        out[t, :k] = v[:k]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="logs/ppo_cec_f1_10d_run.npz",
                        help="Đường dẫn file .npz do train_ppo_cec.py tạo")
    parser.add_argument("--out", type=str, default="logs/ppo_cec_f1_10d_plots.png",
                        help="Đường dẫn ảnh .png để lưu biểu đồ")
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        raise FileNotFoundError(
            f"Không tìm thấy file metrics: {args.npz}\n"
            f"Hãy chạy lại train: python -m scripts.train_ppo_cec --updates 120 "
            f"--seed 2025 --logfile {args.npz}"
        )

    data = np.load(args.npz, allow_pickle=True)

    updates = data["updates"]             # [T]
    alpha_sum = data["alpha_sum"]         # [T]
    loss = data["loss"]                   # [T]
    best = data["best"]                   # [T]
    num_ops = int(data["num_ops"])
    # các trường dạng object: mỗi phần tử là vector (num_ops,)
    alpha_vec = _stack_obj_series(data["alpha_vec"], num_ops)     # [T, num_ops]
    probs = _stack_obj_series(data["probs"], num_ops)             # [T, num_ops]
    op_improve = _stack_obj_series(data["op_improve"], num_ops)   # [T, num_ops]

    # Vẽ
    plt.figure(figsize=(12, 10))

    # 1) alpha_sum và loss (2 trục)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(updates, alpha_sum, label="alpha_sum")
    ax1.set_title("Dirichlet alpha_sum & Loss")
    ax1.set_xlabel("update")
    ax1.set_ylabel("alpha_sum")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(updates, loss, linestyle="--", label="loss")
    ax2.set_ylabel("loss")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 2) probs theo từng operator
    ax3 = plt.subplot(3, 1, 2)
    for j in range(num_ops):
        ax3.plot(updates, probs[:, j], label=f"p{j}")
    ax3.set_title("Dirichlet mean (probs) theo thời gian")
    ax3.set_xlabel("update")
    ax3.set_ylabel("prob")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(ncols=min(4, num_ops), fontsize=8)

    # 3) op_improve theo từng operator
    ax4 = plt.subplot(3, 1, 3)
    for j in range(num_ops):
        ax4.plot(updates, op_improve[:, j], label=f"imp{j}")
    ax4.set_title("Tỉ lệ cải thiện theo operator (op_improve)")
    ax4.set_xlabel("update")
    ax4.set_ylabel("improve rate")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(ncols=min(4, num_ops), fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to: {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
