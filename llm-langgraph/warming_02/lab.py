import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import tiktoken

# loading the document...
file_path = "/home/dusoudeth/Calibre Library/J.K. Rowling/Harry Potter e a Pedra Filosofal (272)/Harry Potter e a Pedra Filosofal - J.K. Rowling.pdf"
raw_documents = PyPDFLoader(file_path=file_path).load()

# splitting the document...
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
chunks = text_splitter.split_documents(raw_documents)

# preparing for embedding the chunks...
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

# indexing the chunks...
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# calculating the number of tokens in each chunk...
def tokens_from_string(string:str, encoding_string:str) -> int:
    encoding = tiktoken.get_encoding(encoding_string)
    return len(encoding.encode(string))

for i, doc in enumerate(chunks):
    print(f"Chunk {i} has {tokens_from_string(doc.page_content, encoding_string="cl100k_base")} tokens")