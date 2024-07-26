from langchain.vectorstores import ElasticsearchStore
from langchain.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.llms.ollama import Ollama

loader = PyPDFDirectoryLoader("C:/Users/kyran/Documents/Python/Elastic/pdf docs")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
print("Splitting Text...")
docs = text_splitter.split_documents(documents)
embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
print("Embedding documents...")
db = ElasticsearchStore.from_documents(
    docs, embedding, es_url="http://localhost:9200", index_name="multiple-pdfs",
)
db.client.indices.refresh(index="multiple-pdfs")
print("Finished.")