
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough
)

load_dotenv()

# Prompt 1: Generate Joke
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Ollama Model
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

parser = StrOutputParser()

# Prompt 2: Explain Joke
prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# Joke Generation Chain
joke_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)

# Parallel Chain
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(
        prompt2,
        model,
        parser
    )
})

# Final Chain
final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

# Invoke Chain
result = final_chain.invoke({
    'topic': 'cricket'
})

# Print Output
print("JOKE:")
print(result['joke'])

print("\nEXPLANATION:")
print(result['explanation'])

# Display Graph
final_chain.get_graph().print_ascii()
