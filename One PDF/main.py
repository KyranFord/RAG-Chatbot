from langchain.chains import RetrievalQA
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.llms.ollama import Ollama

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

db = ElasticVectorSearch(
    elasticsearch_url="http://localhost:9200",
    index_name="test-basic",
    embedding=embedding,
)

model = Ollama(model="phi3:latest")

chain = RetrievalQA.from_chain_type(llm=model, 
                                    chain_type="stuff", 
                                    retriever=db.as_retriever(), 
                                    input_key="question",
                                    verbose=True)
print(chain.invoke("What is The Service Test?"))