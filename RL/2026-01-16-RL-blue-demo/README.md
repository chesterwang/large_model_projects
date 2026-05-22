# 使用 TRL 做 PPO 强化学习微调（toy 示例）

简介
- 这个示例展示如何使用 `trl` 对小型因果语言模型（GPT-2）用 PPO 做强化学习微调（RLHF 风格的简化版）。
- 奖励函数非常简单：如果模型输出包含目标词（默认 "blue"），则给正奖励，否则给负奖励。目的是演示训练流程与 API 使用。

文件
- `requirements.txt` - 所需 Python 包
- `prompts.txt` - 示例 prompts（每行一个 prompt）
- `reward.py` - 奖励计算函数
- `train_ppo.py` - 训练脚本

快速开始（本地 CPU 或单 GPU）
1. 创建并激活虚拟环境（可选）
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   .\venv\Scripts\activate   # Windows

2. 安装依赖
   pip install -r requirements.txt

3. 运行训练（小批量、演示用）
   python train_ppo.py --prompts_file prompts.txt --target_word blue --epochs 2 --batch_size 2

说明与调优建议
- 这是一个 toy 示例，训练设置（模型 gpt2、小 batch）仅供演示。实际 RLHF 需要更复杂的 reward model、更多数据、合适的基线/参考模型和充分的算力。
- 如果使用 GPU，请确保 PyTorch 能访问 GPU（pip 安装的 torch 对应你的 CUDA 版本），并在需要时使用 `accelerate` 启动大规模训练。
- 可将 `reward.py` 中的 reward 函数替换为更复杂的判别/打分模型（例如独立训练的 reward model）。
