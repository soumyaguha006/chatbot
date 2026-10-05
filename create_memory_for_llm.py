from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
	loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
	documents = loader.load()
	return documents
documents=load_pdf_files(data=DATA_PATH)
print("Length of PDF pages: ", len(documents))

# Step 2: Split text into chunks
def create_chunks(extracted_data):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = text_splitter.split_documents(extracted_data)
	return chunks
chunks = create_chunks(documents)
print("Length of text chunks: ", len(chunks))

# Step 3: Create embeddings
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
def create_embeddings():
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
	                                   model_kwargs={"device": "cpu"})
	return embeddings
embedding_model = create_embeddings()

# Step 4: Create vector store
DB_FAISS_PATH="vectorstore/db_faiss"
def create_vector_store(chunks, embedding_model):
	vector_store = FAISS.from_documents(chunks, embedding_model)
	vector_store.save_local(DB_FAISS_PATH)
	return vector_store
vector_store_final = create_vector_store(chunks, embedding_model)
print(f"Vector store saved at: {DB_FAISS_PATH}")
