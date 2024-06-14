# --------------------------------- From T5 Tokenizer invoke

# from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto")

# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))


# --------------------------------- From AutoTokenizer invoke Prefered

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))



# --------------------------------- Errors


# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFacePipeline

# # Load environment variables from .env file
# load_dotenv()

# # Get the API token from environment variables
# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Check if the API token is loaded correctly
# if not api_token:
#     raise ValueError("The Hugging Face API token is not set. Please check your .env file.")

# # Initialize the LangChain model with HuggingFaceEndpoint
# llm = HuggingFacePipeline.from_model_id(
#     model_id="google/flan-t5-base",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )
# print(llm.invoke("What is Deep Learning?"))

# # Example usage
# input_text = "Full form of MS Word"
# response = llm.generate([input_text])  # Pass input_text as a list
# print("Response:", response)




# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint

# load_dotenv()   

# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Initialize the LangChain model with HuggingFaceHub
# llm = HuggingFaceEndpoint(
#     repo_id="google/flan-t5-base",
#     max_new_tokens = 250,
#     huggingfacehub_api_token=api_token
#     )

# # Example usage
# input_text = "Full form of MS Word"
# response = llm.generate([input_text])
# print("Checking ",response)



# from langchain.llms import HuggingFaceHub

# # Initialize the LangChain model with HuggingFaceHub
# llm = HuggingFaceHub(
#     repo_id="google/flan-t5-xxl",
#     model_kwargs={"from_pretrained": "google/flan-t5-xxl"}
# )

# # Example usage
# input_text = "Translate English to French: How are you?"
# response = llm.generate(input_text)
# print(response)


# --------------------------------- ||||||||||||||||||||||||||||


# --------------------------------- Prefered because we did't have to use tokenizer each time on prompt and also did't use encode decode 


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
input_text = "What is full form of MS Word"
response = llm(input_text)
print(response)
