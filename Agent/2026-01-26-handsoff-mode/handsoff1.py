from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from langchain_core.messages import add_messages


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    customer_info: dict
    issue_category: str
    resolution_status: str

def router_node(state):
    # 根据当前状态决定下一个节点
    if state["current_agent"] == "greeting":
        if state["customer_info"]["needs_verification"]:
            return "verification"
        else:
            return "support"
    elif state["current_agent"] == "support":
        if state["issue_category"] == "complex":
            return "expert"
        else:
            return "resolution"
    return "end"

def greeting_node(state):
    # 欢迎客户并收集基本信息
    # 返回更新后的状态
    return {"current_agent": "greeting"}

def verification_node(state):
    # 验证客户身份
    # 返回更新后的状态
    return {"current_agent": "verification"}

def support_node(state):
    # 提供基础支持
    # 返回更新后的状态
    return {"current_agent": "support"}

def expert_node(state):
    # 专家级支持
    # 返回更新后的状态
    return {"current_agent": "expert"}

def resolution_node(state):
    # 解决方案提供
    # 返回更新后的状态
    return {"current_agent": "resolution", "resolution_status": "completed"}

# 构建图
workflow = StateGraph(SupportState)
workflow.add_node("greeting", greeting_node)
workflow.add_node("verification", verification_node)
workflow.add_node("support", support_node)
workflow.add_node("expert", expert_node)
workflow.add_node("resolution", resolution_node)

# 添加条件边
workflow.add_conditional_edges(
    "greeting",
    router_node,
    {
        "verification": "verification",
        "support": "support",
        "end": END
    }
)

# 添加其他条件边
workflow.add_conditional_edges(
    "verification",
    router_node,
    {
        "support": "support",
        "expert": "expert",
        "resolution": "resolution",
        "end": END
    }
)

workflow.add_conditional_edges(
    "support",
    router_node,
    {
        "expert": "expert",
        "resolution": "resolution",
        "end": END
    }
)

workflow.add_conditional_edges(
    "expert",
    router_node,
    {
        "resolution": "resolution",
        "end": END
    }
)

workflow.add_conditional_edges(
    "resolution",
    router_node,
    {
        "end": END
    }
)

# 设置入口点
workflow.set_entry_point("greeting")

# 编译图
app = workflow.compile()