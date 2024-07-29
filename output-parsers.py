#import environment variables
from dotenv import load_dotenv
load_dotenv()

#import model
from langchain_openai import ChatOpenAI
#import prompt template
from langchain_core.prompts import ChatPromptTemplate
#import string output parser
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

#create model
model = ChatOpenAI (
    model = "gpt-3.5-turbo-1106",
    temperature=0.7
    )

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()
    
    chain = prompt | model | parser

    return chain.invoke({
        "input": "dog"
    })
    
def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a list of 10 synomyms for the following word. Return the results as a comma separated list."),
        ("human", "{input}")
    ])
    
    parser = CommaSeparatedListOutputParser()
    
    chain = prompt | model | parser
    
    return chain.invoke({
        "input": "happy"
    })

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])
    
    class Person(BaseModel):
        name: str = Field(description="the name of the person")
        age: int = Field(description="the age of the person")
    
    
    parser = JsonOutputParser(pydantic_object=Person)
    
    chain = prompt | model | parser
    
    return chain.invoke({
        "phrase": "Max is 30 years old",
        "format_instructions": parser.get_format_instructions()
    })
    
#another example

def call_json_output_parser1():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase. \nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])
    
    class Person(BaseModel):
        recipe: str = Field(description="the name of the recipe")
        ingredients: list = Field(description="ingredients")
    
    
    parser = JsonOutputParser(pydantic_object=Person)
    
    chain = prompt | model | parser
    
    return chain.invoke({
        "phrase": "The ingredients for a Margharita pizaa are tomatoes, onions, cheese, basil ",
        "format_instructions": parser.get_format_instructions()
    })
    
# print(type(call_string_output_parser()))
#print(type(call_list_output_parser()))
#print(call_json_output_parser())
print(call_json_output_parser1())


    