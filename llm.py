# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

#instantitate LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo", #type of model latest is gpt-4
    temperature=0.7#how factual -close to zero e.g math, or how creative you want the model to be - close to 1
    #max_tokens=1000, #tokens per generation - charged for the amount of tokens used
    #verbose=True #used to debug the output from the model. Useful for more complex chains  
)

#the invoke method allows one prompt, we are able to pass in multple prompts and get multiple responses back by chaning invoke to 'batch'

#response = llm.invoke("Hello, how are you?")
#print(response)

#response = llm.batch(["Hello, how are you?", "Write a poem about AI"])
#print(response)

response = llm.stream("Write a poem about AI")
#print(response)

for chunk in response:
    print(chunk.content, end="", flush=True)