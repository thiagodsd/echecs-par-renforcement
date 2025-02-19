import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
import tiktoken
from langchain_community.chat_models import ChatMaritalk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# loading the document...
print("loading the document...")
file_path = "/home/dusoudeth/Calibre Library/Joao Ubaldo Ribeiro/Viva o povo brasileiro (249)/Viva o povo brasileiro - Joao Ubaldo Ribeiro.pdf"
raw_documents = PyPDFLoader(file_path=file_path).load()

# splitting the document...
print("splitting the document...")
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
chunks = text_splitter.split_documents(raw_documents)

# preparing for embedding the chunks...
print("preparing for embedding the chunks...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

# indexing the chunks...
print("indexing the chunks...")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# # calculating the number of tokens in each chunk...
# def tokens_from_string(string:str, encoding_string:str) -> int:
#     encoding = tiktoken.get_encoding(encoding_string)
#     return len(encoding.encode(string))

# for i, doc in enumerate(chunks):
#     print(f"Chunk {i} has {tokens_from_string(doc.page_content, encoding_string="cl100k_base")} tokens")

retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":10}
)

prompt = ChatPromptTemplate.from_template(
"""
You are an assistant for question-and-answer sessions about the book "Viva o povo brasileiro" by João Ubaldo Ribeiro.
Use the following pieces of retrieved context to answer the questions.
If you don't know the answer, you can say "I don't know".
QUESTION: {question}
CONTEXT: {context}
ANSWER:
"""
)

model = ChatMaritalk(
    model="sabia-3",
    api_key=os.getenv("MARITACA_KEY"),
)

rag_chain = (
    {
        "context": retriever, 
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

print("\n")
question = "Quem é caboco Capiroba?"
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n")