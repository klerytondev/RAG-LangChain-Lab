import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser


def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters() 

# Carregamento do documento PDF
pdf_path = 'data/laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Divisão do texto em pedaços
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Criação da cadeia de processamento para dividir o texto em pedaços
chunks = text_splitter.split_documents(
    documents=docs,
)

# Criação do vetorizador
embedding = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name='laptop_manual',
)

# Criação do recuperador
retriever = vector_store.as_retriever()

# Prompt para a IA se comportar como um assistente de pesquisa usando o RAG
prompt = hub.pull('rlm/rag-prompt')

# Cadeia de processamento para responder perguntas
rag_chain = (
    {
        'context': retriever,
        'question': RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

try:
    while True:
        question = input('Qual a sua dúvida? ')
        response = rag_chain.invoke(question)
        print(response)
except KeyboardInterrupt:
    exit()
