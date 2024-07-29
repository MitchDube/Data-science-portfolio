#import environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
#import prompt template
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore): 
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt 
    )
    
    #Retrieve the most relavant documents from the vector store
    #need to create a history aware retriever
    #retriever = vectorStore.as_retriever()
    retriever = vectorStore.as_retriever(search_kwargs={"k": 7})
    
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history")
        ("human", "{input}"),
        ("humam", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )
    
    retrieval_chain = create_retrieval_chain(
        #retriever,
        history_aware_retriever,
        chain
    )
    
    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
    "input": question,
    "chat_history": chat_history

    })
    return response["answer"]

# Ensures that this code will execute of the this is the main file that is being processed and not if it is imported by another file
""""
Using if __name__ == '__main__': is a common Python idiom that allows code to be run when the script is executed directly but not when 
it is imported as a module in another script. This is useful for testing, running scripts, and keeping the code organized.
"""
if __name__ == '__main__':
    docs = get_documents_from_web('https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response/?gad_source=1&gclid=CjwKCAjw2Je1BhAgEiwAp3KY7yKLdi9tKVae1Wtii5v5Voc3P3PjneCBubdCxRL2RRZV2UV8XmTINhoCFbEQAvD_BwE')
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)
    
    #create conversation history
    chat_history = []
    
    while True:    
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        
        print("Assistant:", response)


