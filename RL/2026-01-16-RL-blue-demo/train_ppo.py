"""
简化的 PPO 训练脚本（toy）。
用法示例：
python train_ppo.py --prompts_file prompts.txt --target_word blue --epochs 2 --batch_size 2
"""
import argparse
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from reward import simple_contains_reward

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

def chunkify(lst, n):
    """yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, default="prompts.txt")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--target_word", type=str, default="blue")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(args.prompts_file)
    if len(prompts) == 0:
        raise ValueError("prompts file is empty")

    # 配置 PPO
    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=1.41e-5,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,  # 小示例直接等于 batch_size
    )

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # GPT2 没有 pad token，设置为 eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # 创建参考模型（ref_model 在 PPO 中用于估计 KL）
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # 创建 PPOTrainer
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer)

    # 训练循环（每次从 prompts 中采样小批量）
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        random.shuffle(prompts)
        for batch_prompts in chunkify(prompts, args.batch_size):
            # 生成 responses（使用 PPOTrainer 的 generate 方法）
            # 这里我们使用 trainer.generate 接口（trl 版本间可能略有不同）
            # generate 返回 list[str]
            responses = ppo_trainer.generate(batch_prompts, max_new_tokens=args.max_new_tokens)

            # 计算奖励
            rewards = [simple_contains_reward(r, args.target_word) for r in responses]

            # PPO 更新一步：传入 queries (prompts), responses, rewards
            # trainer.step 会处理 tokenization 与梯度更新
            stats = ppo_trainer.step(batch_prompts, responses, rewards)

            # 打印信息
            avg_reward = sum(rewards) / len(rewards)
            print(f"  batch_size={len(batch_prompts)} avg_reward={avg_reward:.3f} loss_stats={stats}")

    # 保存微调后的模型
    model.save_pretrained("ppo_finetuned_model")
    tokenizer.save_pretrained("ppo_finetuned_model")
    print("Training complete. Model saved to ./ppo_finetuned_model")

if __name__ == "__main__":
    main()