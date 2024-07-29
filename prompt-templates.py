from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate 

#Instantiate Model
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo-1106"
)

#Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the results to a comma separated list"),
        ("human", "{input}")
    ]
)

#Create llm chain (the output from the prompt will be the input to the llm)
chain = prompt | llm

response = chain.invoke({"input": "happy"})
print(type(response))
    
#output_list = [item.strip() for item in response.split(',')]
#print(f"Output List: {response}")
