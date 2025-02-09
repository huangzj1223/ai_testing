from abc import abstractmethod
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR, StorageContext

class BaseRAG:
    def __init__(self, files: list[str]):
        self.files = files
    @abstractmethod
    async def load_data(self):
        """加载数据"""
    async def create_local_index(self, persist_dir=DEFAULT_PERSIST_DIR) -> BaseIndex:
        """
        创建本地索引，该函数是数据嵌入的重点优化模块
        入库优化：数据清洗优化--》分块优化
        参考LLmaindex的分块策略：https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/
        :param persist_dir: 本地持久化路径
        :return: BaseIndex
        """
        # 加载数据
        data = await self.load_data()
        # 创建一个句子分割器
        node_splitter = SentenceSplitter.from_defaults(separator="。")
        # 从文档中获取节点
        nodes = node_splitter.get_nodes_from_documents(data, show_progress=True)
        # 创建向量存储索引，该部分需要用到嵌入模型，当前嵌入模型的设置在utils/settings.py中

        index = VectorStoreIndex(nodes, show_progress=True)
        # index = VectorStoreIndex.from_documents(data, show_progress=True)
        # 对向量数据库做持久化
        index.storage_context.persist(persist_dir=persist_dir)
        # 返回创建的索引
        return index

    async def create_remote_index(self, collection_name="default") -> BaseIndex:
       pass

    @staticmethod
    async def load_remote_index(collection_name="default") -> BaseIndex:
        pass

    @staticmethod
    async def load_local_index(persist_dir=DEFAULT_PERSIST_DIR) -> BaseIndex:
        return load_index_from_storage(
            StorageContext.from_defaults(persist_dir=persist_dir)
        )

