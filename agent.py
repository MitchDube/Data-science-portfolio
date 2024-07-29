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
from langchain.chains import LLMChain #to accomodate the memory object in our chain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

# Create Retriever
loader = WebBaseLoader('https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response/?gad_source=1&gclid=CjwKCAjw2Je1BhAgEiwAp3KY7yKLdi9tKVae1Wtii5v5Voc3P3PjneCBubdCxRL2RRZV2UV8XmTINhoCFbEQAvD_BwE')
docs = loader.load()
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)
    
embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(docs, embedding=embedding)
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

search = TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    'lcel_search'
    "Use this tool when searching for information about Langchain Expression Language (LCEL)"
)
tools = [search, retriever_tools]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools,
)

#use agent executor to invoke our agent
AgentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
    "input": user_input,
    "chat_history": chat_history
    })

    return response["output"]

if __name__ == '__main__':  
    #create conversation history
    chat_history = []
    
    while True:    
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = process_chat(AgentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        
        print("Assistant:", response)