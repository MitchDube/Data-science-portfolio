#import environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain #to accomodate the memory object in our chain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

#Will be added to environment variables for confidentiality
UPSTASH_URL="https://verified-piranha-58061.upstash.io"
UPSTASH_TOKEN="AeLNAAIjcDE5MTNmNTkwMDRlY2Y0NTA2YTNjOWFiOTFlMGI4OWI3YnAxMA"

history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN,
    session_id= "Chat1", #unique identifier for this conversation. Used to retrieve the conversation from the database so that we can continue with the conversation.
    ttl= 0 #timed in seconds 0 means conversation will never expire, 60 means it expires in 60 seconds
)

model = ChatOpenAI(
    model ="gpt-3.5-turbo",
    temperature=0.7
)

prompt = ChatPromptTemplate([
    ("system", "You are a friendly AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

# chain = memory | prompt | model 
#This gives an error because this memory object is not a runnable. Hence we use a class to create a chain 
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True  
)

#create separate variable for our message. "hello" needs to match with every unput from the promt variable
#msg1 = {
#    "input": "What is mental health"
#}

#response1 = chain.invoke(msg1)
#print(response1)

msg2 = {
    "input": "How can I cope with stress?"
}

response2 = chain.invoke(msg2)
print(response2)