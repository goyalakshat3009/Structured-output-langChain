
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Ollama model
model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

# LCEL Chain
chain = template1 | model | parser | template2 | model | parser

# Invoke chain
result = chain.invoke({"topic": "black hole"})

print(result)
