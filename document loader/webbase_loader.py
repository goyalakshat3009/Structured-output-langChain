
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Ollama Model
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

# Prompt
prompt = PromptTemplate(
    template="""Answer the following question:
{question}

Use the following text:
{text}""",
    input_variables=['question', 'text']
)

# Output Parser
parser = StrOutputParser()

# Website URL
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'

# Load Website
loader = WebBaseLoader(url)

docs = loader.load()

print("Total documents:", len(docs))

# Create Chain
chain = prompt | model | parser

# Invoke Chain
result = chain.invoke({
    'question': 'What is the product that we are talking about?',
    'text': docs[0].page_content
})

print(result)
