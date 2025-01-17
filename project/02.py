import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser

# Defini a função que inicializa os parâmetros necessários
def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters() 

# Criação do banco de dados
pdf_path = 'data/laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Criação do toolkit
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(
    documents=docs,
)

# Criação do agente
persist_directory = 'db'

# Criação do executor
embedding = OpenAIEmbeddings()

# Criação do Chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name='laptop_manual',
)
