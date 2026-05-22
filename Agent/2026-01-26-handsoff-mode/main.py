import os
import uuid
from typing import Callable, Literal
from typing_extensions import NotRequired

from langchain_community.chat_models import ChatTongyi
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# =====================
# 0) 模型初始化（ChatTongyi）
# =====================
model = ChatTongyi(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    model="qwen-max",
    temperature=0.3,
)

# =====================
# 1) 自定义 State
# =====================
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """客服流程的 state。"""

    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]

# =====================
# 2) 工具：更新 state + 推进阶段
# =====================
@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录保修状态，并推进到问题分类阶段。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态已记录：{status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )

@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录问题类型，并推进到解决方案阶段。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型已记录：{issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """升级到人工支持。"""
    return f"已为你升级到人工支持。原因：{reason}"

@tool
def provide_solution(solution: str) -> str:
    """提供解决方案。"""
    return f"已提供解决方案：{solution}"

# 可选：回退
@tool
def go_back_to_warranty() -> Command:
    """回到保修确认阶段。"""
    return Command(update={"current_step": "warranty_collector"})

@tool
def go_back_to_classification() -> Command:
    """回到问题分类阶段。"""
    return Command(update={"current_step": "issue_classifier"})

# =====================
# 3) 每个阶段的 prompt + tools 配置
# =====================
WARRANTY_COLLECTOR_PROMPT = """你是一名售后客服助手，正在帮助用户解决设备问题。

【当前阶段：确认保修】
你需要：
1. 友好地问候用户
2. 询问设备是否仍在保修期（或询问购买时间/订单信息用于判断）
3. 一旦信息足够明确，必须调用 record_warranty_status 记录结果并进入下一阶段

要求：语气自然、友好，不要一次问太多问题。"""

ISSUE_CLASSIFIER_PROMPT = """你是一名售后客服助手，正在帮助用户解决设备问题。

【当前阶段：问题分类】
已知信息：保修状态 = {warranty_status}

你需要：
1. 引导用户描述问题现象
2. 判断问题属于【硬件】还是【软件】
3. 一旦判断足够明确，必须调用 record_issue_type 记录分类并进入下一阶段

如果不明确，可以继续追问，但不要武断下结论。"""

RESOLUTION_SPECIALIST_PROMPT = """你是一名专业的售后支持工程师，正在帮助用户解决设备问题。

【当前阶段：给出解决方案】
已知信息：
- 保修状态 = {warranty_status}
- 问题类型 = {issue_type}

你需要：
1. 如果是【软件问题】：调用 provide_solution 给出清晰的排查/修复步骤（从低风险到高风险）
2. 如果是【硬件问题】：
   - 在保修期：调用 provide_solution 说明官方保修维修流程、备份与注意事项
   - 不在保修期：调用 escalate_to_human 转人工说明付费维修选择

如果用户纠正了信息：
- 用 go_back_to_warranty 回到保修确认
- 用 go_back_to_classification 回到问题分类

要求：回复具体、可执行、条理清晰。"""

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human, go_back_to_warranty, go_back_to_classification],
        "requires": ["warranty_status", "issue_type"],
    },
}

# =====================
# 4) Middleware：按 current_step 动态切换配置
# =====================
@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    current_step = request.state.get("current_step", "warranty_collector")
    stage_config = STEP_CONFIG[current_step]

    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"在进入 {current_step} 之前，必须先设置 {key}")

    system_prompt = stage_config["prompt"].format(**request.state)

    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)

# =====================
# 5) 创建 Agent（注册所有工具 + checkpointer）
# =====================
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
    go_back_to_warranty,
    go_back_to_classification,
]

agent = create_agent(
    model=model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

# =====================
# 6) 测试
# =====================
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=== Turn 1: 用户报问题 ===")
    r1 = agent.invoke({"messages": [HumanMessage("你好，我的手机屏幕摔裂了")]}, config)
    for m in r1["messages"]:
        m.pretty_print()

    print("=== Turn 2: 用户回答保修 ===")
    r2 = agent.invoke({"messages": [HumanMessage("还在保修期，去年买的")]}, config)
    for m in r2["messages"]:
        m.pretty_print()

    print("=== Turn 3: 用户描述现象 ===")
    r3 = agent.invoke({"messages": [HumanMessage("裂纹很明显，而且触摸不太灵敏")]}, config)
    for m in r3["messages"]:
        m.pretty_print()

    print("=== Turn 4: 询问怎么处理 ===")
    r4 = agent.invoke({"messages": [HumanMessage("我现在应该怎么处理？")]}, config)
    for m in r4["messages"]:
        m.pretty_print()