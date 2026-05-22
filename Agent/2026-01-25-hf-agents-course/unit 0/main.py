from smolagents import LiteLLMModel
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

def main():
    model = LiteLLMModel(
        model_id="deepseek-ai/DeepSeek-V3.2",  # Or try other Ollama-supported models
        api_base=os.getenv("SILICONFLOW_BASE_URL"),  # Default Ollama local server
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        num_ctx=8192,
    )

    model

if __name__ == "__main__":
    main()