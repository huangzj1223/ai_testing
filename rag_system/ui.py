from typing import Optional, List
import chainlit as cl
from chainlit.element import ElementBased
from chainlit.types import ThreadDict
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from rag.document import DocumentRAG
import chainlit.data as cl_data

# 初始化聊天记录存储持久层
from utils import settings
from persistent.minio_storage_client import MinioStorageClient
from rag_system.persistent.postgresql_data_layer import PostGreSQLDataLayer
storage_client = MinioStorageClient()
cl_data._data_layer = PostGreSQLDataLayer(conninfo=settings.configuration.pg_connection_string, storage_provider=storage_client)

async def view_pdf(elements: List[ElementBased]):
    """查看PDF文件"""
    files = []
    contents = []
    for element in elements:
        if element.name.endswith(".pdf"):
            pdf = cl.Pdf(name=element.name, display="side", path=element.path)
            files.append(pdf)
            contents.append(element.name)
    if len(files) == 0:
        return
    await cl.Message(content=f"查看PDF文件：" + "，".join(contents), elements=files).send()


@cl.on_chat_start
async def start():
    # 初始化默认聊天引擎（直接与大模型对话）
    memory = ChatMemoryBuffer.from_defaults(token_limit=1024)
    chat_engine = SimpleChatEngine.from_defaults(memory=memory)
    cl.user_session.set("chat_engine", chat_engine)

@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    profiles = [
        cl.ChatProfile(
            name="大模型对话",
            markdown_description=f"与大模型对话",
            icon=f"/public/kbs/model.png",
        ),
        cl.ChatProfile(
            name="数据库对话",
            markdown_description=f"与数据库对话",
            icon=f"/public/kbs/db.jpg",
        ),
        cl.ChatProfile(
            name="EXCEL对话",
            markdown_description=f"与Excel对话",
            icon=f"/public/kbs/excel.png",
        )

    ]

    return profiles

@cl.set_starters
async def set_starters():
    starters = [
        cl.Starter(
            label="大模型提高软件测试效率",
            message="详细介绍如何借助大语言模型提高软件测试效率。",
            icon="/public/apidog.svg",
        ),
        cl.Starter(
            label="自动化测试思路",
            message="详细描述一下接口及UI自动化测试的基本思路。",
            icon="/public/pulumi.svg",
        ),
        cl.Starter(
            label="性能测试分析及瓶颈定位思路",
            message="详细描述一下软件性能测试分析及瓶颈定位的核心思路。",
            icon="/public/godot_engine.svg",
        )
    ]

    return starters

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    # 可以对接第三方认证
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin",
                       metadata={"role": "admin", "provider": "credentials"})
    else:
        return None


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    chat_engine = SimpleChatEngine.from_defaults()
    for message in thread.get("steps", []):
        if message["type"] == "user_message":
            chat_engine.chat_history.append(ChatMessage(content=message["output"], role="user"))
        elif message["type"] == "assistant_message":
            chat_engine.chat_history.append(ChatMessage(content=message["output"], role="assistant"))
    cl.user_session.set("chat_engine", chat_engine)

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="", author="Assistant")
    chat_mode = cl.user_session.get("chat_profile", "大模型对话")

    if chat_mode == "大模型对话":
        await view_pdf(message.elements)
        files = []
        # 获取用户上传的文件（包含图片）
        for element in message.elements:
            if isinstance(element, cl.File) or isinstance(element, cl.Image):
                files.append(element.path)

        if len(files) > 0:
            rag = DocumentRAG(files=files)
            index = await rag.create_local_index()
            chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT)
            cl.user_session.set("chat_engine", chat_engine)

    chat_engine = cl.user_session.get("chat_engine")
    res = await cl.make_async(chat_engine.stream_chat)(message.content)

    # 流式界面输出
    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()
