from langchain.vectorstores import ElasticsearchStore
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.llms.ollama import Ollama
####### Load the document into  elastic store using hugging face embeddings as the embedding function. Which could be one time activity depending on the use-case.

loader = PyPDFLoader("C:/Users/kyran/Documents/Python/pdfs/armyvalues.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

db = ElasticsearchStore.from_documents(
    docs, embedding, es_url="http://localhost:9200", index_name="test-basic",
)

db.client.indices.refresh(index="test-basic")