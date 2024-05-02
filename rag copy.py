import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama 

data_path = "./doc.pdf"
text_splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
    length_function=len,)

text_splitter2 = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter1)

#embedding_func = OllamaEmbeddings(model="nomic-embed-text:latest")
embedding_func = OllamaEmbeddings(model="nomic-embed-text:latest")
print("embedding")
vectordb = Chroma.from_documents(documents, embedding=embedding_func, persist_directory="./embeddingsarmy")
#vectordb = Chroma(persist_directory="./embeddingsarmy", embedding_function=embedding_func)
template = """<s>[INST] Only generate prompts from context give - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]"""
pt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

rag = RetrievalQA.from_chain_type(
            llm=Ollama(model="mistral:7b-instruct-v0.2-q5_K_M", num_ctx = 32768),
            retriever=vectordb.as_retriever(),
            memory=ConversationSummaryMemory(llm = Ollama(model="mistral:7b-instruct-v0.2-q5_K_M")),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )
print(rag.invoke("Question"))