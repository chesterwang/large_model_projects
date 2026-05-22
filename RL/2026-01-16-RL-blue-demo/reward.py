import re

def simple_contains_reward(response: str, target_word: str = "blue") -> float:
    """
    简单的奖励函数：如果 response 中包含 target_word (case-insensitive)，返回 +1，否则返回 -1。
    你可以把它换成更复杂的 reward model（例如用一个单独训练的判别模型给分）。
    """
    if not isinstance(response, str):
        return -1.0
    # 简单的词边界检测（忽略大小写）
    if re.search(rf"\\b{re.escape(target_word)}\\b", response, flags=re.IGNORECASE):
        return 1.0
    return -1.0