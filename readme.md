# Invoking LLM Models: Methods and Best Practices

This documentation provides a detailed overview of different methods to invoke and use Large Language Models (LLMs) with examples. It also covers additional features like retrieval-based question answering and memory management using LangChain.

## Table of Contents
1. [Using T5 Tokenizer](#using-t5-tokenizer)
2. [Using AutoTokenizer (Preferred)](#using-autotokenizer-preferred)
3. [Using HuggingFace Pipeline with LangChain](#using-huggingface-pipeline-with-langchain)
4. [Handling Errors and API Tokens](#handling-errors-and-api-tokens)
5. [RetrievalQA](#retrievalqa)
6. [Memory Management](#memory-management)
    - [MemoryWindow](#memorywindow)
    - [MemoryBufferWindow](#memorybufferwindow)
7. [Differences Between Methods](#differences-between-methods)
8. [References](#references)

## Using T5 Tokenizer

The T5 Tokenizer method involves directly using the `T5Tokenizer` and `T5ForConditionalGeneration` classes from the `transformers` library. This method requires encoding the input text into token IDs and then generating the output. The generated token IDs are then decoded back into text. This approach gives you control over the tokenization process but can be a bit verbose.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))


Using AutoTokenizer (Preferred)
The AutoTokenizer method utilizes AutoTokenizer and AutoModelForSeq2SeqLM from the transformers library. It is similar to the T5 Tokenizer method but offers a more flexible and unified interface that can automatically select the appropriate model and tokenizer. This method simplifies the process and is generally preferred for its ease of use and adaptability to different models.


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))


Using HuggingFace Pipeline with LangChain
This method involves using the pipeline function from the transformers library to create a text generation pipeline. The pipeline is then wrapped with HuggingFacePipeline from LangChain to facilitate integration with LangChain's functionality. This approach abstracts away the tokenization and decoding steps, making it more straightforward and efficient for generating text.

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the tokenizer and model using transformers
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Create a text generation pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250)

# Initialize LangChain with the HuggingFace pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Example usage
input_text = "What is the full form of MS Word"
response = llm(input_text)
print(response)


Handling Errors and API Tokens
Managing API tokens and handling potential errors is crucial when working with cloud-based or API-dependent models. This section explains how to load environment variables using the dotenv library, check for the presence of required API tokens, and initialize models using HuggingFacePipeline. Proper error handling ensures that the application runs smoothly and securely.

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline

# Load environment variables from .env file
load_dotenv()

# Get the API token from environment variables
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Check if the API token is loaded correctly
if not api_token:
    raise ValueError("The Hugging Face API token is not set. Please check your .env file.")

# Initialize the LangChain model with HuggingFaceEndpoint
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

# Example usage
input_text = "Full form of MS Word"
response = llm.generate([input_text])  # Pass input_text as a list
print("Response:", response)


RetrievalQA
Retrieval-based Question Answering (RetrievalQA) involves retrieving relevant documents or context before generating an answer. This method enhances the accuracy and relevance of responses by leveraging a vector store (such as FAISS) and embeddings (such as those from Hugging Face). RetrievalQA chains combine these retrieval capabilities with language model generation to provide more contextually appropriate answers.

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize vector store and embeddings
vector_store = FAISS(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
retriever = vector_store.as_retriever()

# Initialize RetrievalQA chain
qa_chain = RetrievalQA(retriever=retriever, llm=llm)

# Example usage
input_question = "What is the capital of France?"
response = qa_chain.run(input_question)
print(response)


Memory Management
LangChain provides mechanisms to manage memory within conversations, maintaining context over multiple interactions. This feature is essential for building conversational agents that need to remember past exchanges.

MemoryWindow
MemoryWindow keeps a fixed-size window of recent interactions. This approach helps in managing the context window by retaining only the most recent interactions, which is useful for scenarios where the conversation history is not too long.

from langchain.memory import WindowMemory

# Initialize WindowMemory with a specific window size
memory = WindowMemory(window_size=5)

# Example usage
memory.add_context("User", "What is AI?")
memory.add_context("Assistant", "AI stands for Artificial Intelligence.")
response = memory.get_context()
print(response)

MemoryBufferWindow
MemoryBufferWindow maintains a buffer of recent interactions, which can be useful for keeping a more extensive history of the conversation. This method is beneficial when the context needs to include more information from past interactions, providing a richer context for the model to generate responses.

from langchain.memory import MemoryBufferWindow

# Initialize MemoryBufferWindow with a specific buffer size
memory = MemoryBufferWindow(buffer_size=5)

# Example usage
memory.add_context("User", "Tell me about Python.")
memory.add_context("Assistant", "Python is a high-level programming language.")
response = memory.get_context()
print(response)

Differences Between Methods
Control and Flexibility
T5 Tokenizer: Provides more control over the tokenization and decoding process but requires more boilerplate code.
AutoTokenizer: Simplifies the process with a more flexible and unified interface, automatically selecting the appropriate model and tokenizer.
HuggingFace Pipeline with LangChain: Abstracts away tokenization and decoding, offering a streamlined and efficient approach for text generation.
Ease of Use
AutoTokenizer and HuggingFace Pipeline methods are generally easier to use and more user-friendly, especially for those who prefer not to handle tokenization explicitly.
Integration
HuggingFace Pipeline with LangChain: This method integrates seamlessly with LangChain, enabling additional features like memory management and RetrievalQA.
Memory Management
MemoryWindow: Suitable for managing a short history of interactions.
MemoryBufferWindow: Useful for maintaining a more extensive conversation history.
Choosing the appropriate method depends on the specific requirements of the project, including the need for control, ease of use, and integration with other functionalities like retrieval and memory management.

References
Transformers Documentation
LangChain Documentation
Hugging Face API
Streamlit
Dotenv