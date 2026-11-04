
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)

load_dotenv()

# Function to count words
def word_count(text):
    return len(text.split())


# Prompt
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Ollama Model
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

parser = StrOutputParser()

# Joke Generation Chain
joke_gen_chain = RunnableSequence(
    prompt,
    model,
    parser
)

# Parallel Chain
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

# Final Chain
final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

# Invoke
result = final_chain.invoke({'topic': 'AI'})

# Final Output
final_result = """{} 
word count - {}""".format(
    result['joke'],
    result['word_count']
)

print(final_result)

# Graph
final_chain.get_graph().print_ascii()
