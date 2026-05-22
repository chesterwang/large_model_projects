import asyncio
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-99LcwQMmtkMui0K5kzzZ932gh7eqNVCVib7u6Jd1dC431stm"
os.environ["ANTHROPIC_BASE_URL"] = "https://api.moonshot.cn/anthropic/"
os.environ["ANTHROPIC_MODEL"] = "kimi-k2-thinking-turbo"

from claude_agent_sdk import query, ClaudeAgentOptions

# os.environ['ANTHROPIC_API_KEY'] =  "sk-8cbd7d1f9aef4b408ade7d9c66481e03"
# os.environ['ANTHROPIC_BASE_URL'] =  "https://dashscope.aliyuncs.com/compatible-mode/v1"


async def main():
    async for message in query(
        prompt="为什么",
        options=ClaudeAgentOptions(
            allowed_tools=["Bash", "Glob"], model=os.environ.get("ANTHROPIC_MODEL"),
            cwd="/tmp/ppp"
        ),
    ):
        if hasattr(message, "result"):
            print("----------------result----------------")
            print(message.result)
        else:
            print("----------------step No. X----------------")
            print(message)




if __name__ == "__main__":
    asyncio.run(main())
