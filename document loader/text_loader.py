
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Groq Model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Prompt
prompt = PromptTemplate(
    template='Write a summary for the following poem:\n{poem}',
    input_variables=['poem']
)

# Output Parser
parser = StrOutputParser()

# Load Text File
loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

# Document Information
print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

# Chain
chain = prompt | model | parser

# Final Output
print(chain.invoke({
    'poem': docs[0].page_content
}))
