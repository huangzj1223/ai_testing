from llama_index.core import SimpleDirectoryReader, Document

from rag.base import BaseRAG


class DocumentRAG(BaseRAG):

    async def load_data(self):
        """
        加载数据，该函数需要优化文件内容的识别、清洗
        :return:
        """
        docs = SimpleDirectoryReader(input_files=self.files).load_data()
        # docs = []
        # for file in self.files:
        #     data = SimpleDirectoryReader(input_files=[file]).load_data()
        #     doc = Document(text="\n\n".join([d.text for d in data[0:]]), metadata={"path": file})
        #     docs.append(doc)
        return docs
