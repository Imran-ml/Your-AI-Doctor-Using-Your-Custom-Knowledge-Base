import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StdOutCallbackHandler

def create_index():
    # Load the data from CSV file
    data_loader = CSVLoader(file_path="data.csv")
    data = data_loader.load()
    
    # Create the embeddings model
    embeddings_model = OpenAIEmbeddings()

    # Create the cache backed embeddings in vector store
    store = LocalFileStore("./cache")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_model, store, namespace=embeddings_model.model
    )
    
    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(data, embeddings_model)

    return vector_store.as_retriever()

def setup_openai(openai_key):
    # Set the API key for OpenAI
    os.environ["OPENAI_API_KEY"] = openai_key
    
    # Create index retriever
    retriever = create_index()
    
    # Initialize ChatOpenAI model
    chat_openai_model = ChatOpenAI(temperature=0)
    
    return retriever, chat_openai_model

def ai_doctor_chat(openai_key, query):
    # Setup OpenAI environment
    retriever, chat_model = setup_openai(openai_key)
    
    # Create the QA chain
    handler = StdOutCallbackHandler()
    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        callbacks=[handler],
        return_source_documents=True
    )

    # Ask a question/query
    res = qa_with_sources_chain({"query": query})
    return res['result']
