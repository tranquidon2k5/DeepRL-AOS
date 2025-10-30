# DeepRL-AOS: Adaptive Operator Selection with PPO + DE/PSO Baselines

## Mục đích dự án
Dự án tập trung ứng dụng reinforcement learning (PPO) cho bài toán Adaptive Operator Selection (AOS) trong tối ưu hóa hàm liên tục, dùng các benchmark hàm CEC2017 và các baseline tiến hóa cổ điển (DE, PSO). Mục tiêu là tự động hóa quá trình chọn toán tử tối ưu trong Differential Evolution (DE) bằng một agent RL.

---

## Cấu trúc thư mục dự án

- **algos/**: 
  - `de_baseline.py`, `pso_baseline.py`: Các thuật toán DE, PSO baseline chuẩn để đối chứng.
- **envs/**:
  - `aos_gym.py`: Môi trường Gymnasium cho bài toán tối ưu, hỗ trợ state chuẩn hóa, action space Dirichlet, reward và thống kê operator.
- **rl/**:  
  - `ppo.py`: Định nghĩa agent PPO (`PPOAgent`), cấu hình (`PPOConfig`), và các hàm collect, update batch RL.
  - `nets.py`: Actor-Critic network. Gồm DeepSetsEncoder để encode quần thể, policy head cho phân phối Dirichlet, Critic head.
- **ops/**:
  - `de_ops.py`: Toàn bộ DE operators: `de_rand_1_bin`, `de_best_2_bin`, `de_current_to_pbest_1`, `gaussian_mutation` và dispatcher.
- **utils/**:
  - `eval.py`: Hàm batch-safe fitness evaluation.
- **scripts/**: Các script train/test.
  - `train_ppo_cec.py`: Main script train PPO agent với Gym env.
  - `run_pso_cec.py`, `eval_compare.py`, `plot_metrics.py`...
  - ...
- **conf/**: Cấu hình YAML cho hyperparams, workflow.
- **checkpoints/**, **logs/**, **plots/**, **runs/**, **reports/**: Output/biểu đồ/thống kê thử nghiệm.

---

## Workflow & Hướng dẫn sử dụng

### Cài đặt:
```bash
pip install -r requirements.txt
```
*Yêu cầu: Python >=3.8, PyTorch, Gymnasium* và các package trong `requirements.txt`.

### Train PPO (ví dụ benchmark CEC2017):
```bash
python scripts/train_ppo_cec.py --config conf/config_best.yaml --updates 120 --function_id 1 --dim 10 --pop 50
```

### Chạy baseline DE/PSO:
```bash
python scripts/run_pso_cec.py  # hoặc kiểm thử với algos/de_baseline.py, algos/pso_baseline.py
```


---

## Chi tiết module chính

### scripts/train_ppo_cec.py
- Nhận argpaser và config từ file YAML.
- Init môi trường (AOSGym) & PPO agent.
- Thiết lập các hyperparams: Dirichlet policy, reward anneal, history logs.
- Vòng lặp updates: lấy sample batch -> agent update policy -> log/tracking

### envs/aos_gym.py (class AOSGym)
- Chuẩn hóa state cho quần thể, action dạng vector phân phối xác suất (Dirichlet cho PPO).
- Tính reward: kết hợp cải tiến log và diverse, có penalty stagnation & cơ chế reset cá thể kẹt.
- Lưu thống kê action/op cho logging/training.

### rl/ppo.py (class PPOAgent)
- `collect`: Thu thập trajectory của agent với môi trường.
- `update`: PPO logic gồm policy gradient, GAE, entropy bonus, gradient clipping.
- PPOConfig: Toàn bộ hyperparams quan trọng

### rl/nets.py
- DeepSetsEncoder: Nhúng trạng thái quần thể dạng permutation-invariant.
- ActorCritic: Policy head Dirichlet, Value head critic. Có hỗ trợ prior bonus, anneal tổng alpha, sharpen.

### algos/de_baseline.py, pso_baseline.py
- Định nghĩa truyền thống cho các thuật toán tiến hóa baseline.

### ops/de_ops.py
- Định nghĩa toàn bộ operator DE đủ loại và dispatcher để áp dụng trên từng cá thể.

### utils/eval.py
- `eval_pop_safe`: Đánh giá batch hoặc từng điểm, đảm bảo output giống CEC.

### conf/
- Lưu các hyperparam tập trung để tái lập thí nghiệm/lập script pipeline.

---

## Dành cho Developer & ChatGPT
- Có thể mở rộng code sang LLM hoặc diffusion model-based AOS theo cấu trúc module đã chuẩn.
- Toàn bộ pipeline logging, checkpoint, modular, kiểu rõ ràng.
- Nên dùng IDE có debug/test để kiểm tra reward/state/action trong env dễ dàng.

---

## Liên hệ / Đóng góp
- Pull request/issue qua GitHub!
- Email maintainer nếu cần hướng dẫn mở rộng nghiên cứu (diffusion models/LLMs, RL giải quyết AOS, ...)!
