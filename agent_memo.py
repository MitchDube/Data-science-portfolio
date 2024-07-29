#import environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain #to accommodate the memory object in our chain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

import os

# Will be added to environment variables for confidentiality
UPSTASH_URL = os.getenv("UPSTASH_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_TOKEN")

history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN,
    session_id="Chat1", # unique identifier for this conversation. Used to retrieve the conversation from the database so that we can continue with the conversation.
    ttl=0 # timed in seconds 0 means conversation will never expire, 60 means it expires in 60 seconds
)

# Create Retriever
loader = WebBaseLoader('https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response/?gad_source=1&gclid=CjwKCAjw2Je1BhAgEiwAp3KY7yKLdi9tKVae1Wtii5v5Voc3P3PjneCBubdCxRL2RRZV2UV8XmTINhoCFbEQAvD_BwE')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(splitDocs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Mitchel."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

search = TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    description='mental_health_search',
    name='mental_health_awareness',
    #usage="Use this tool when searching for information about mental health"
)
tools = [search, retriever_tools]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

# Use AgentExecutor to invoke our agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)

def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

if __name__ == '__main__':
    # Create conversation history
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agent_executor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)