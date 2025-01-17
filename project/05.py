import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser

# Função que inicializa os parâmetros necess
def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters() 

# Criação do banco de dados
db = SQLDatabase.from_uri('sqlite:///ipca.db')

# Criação do toolkit
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)

# Puxa pormpt do hub para o agente
system_message = hub.pull('hwchase17/react')

# Criação do agente
agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

# Criação do executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
Use as ferrmentas necessárias para responder perguntas relacionadas ao histórico de IPCA ao longo dos anos.
Responda tudo em português brasileiro.
Perguntas: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

question = '''
Baseado nos dados históricos de IPCA desde 2004,
faça uma previsão dos valores de IPCA de cada mês futuro até o final de 2024.
'''

# Invoque o agente
output = agent_executor.invoke({
    'input': prompt_template.format(q=question)
})

print(output.get('output'))
