import os
import textwrap

import langchain
import chromadb
import transformers
import openai
import torch

from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from huggingface_hub import login
login(token='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')

# CONFIG
# OpenAI embedding model
os.environ["OPENAI_API_KEY"] = "sk-proj-HQ4mTWV22vFDUzUah986T3BlbkFJGQ42tgoYZyXNwBE6SqJH"

# set up HuggingFace Pipeline with Llama-2-7b-chat-hf model
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
      "text-generation", #task
      model=model,
      tokenizer=tokenizer,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      device_map="auto",
      max_length=1000,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id
)

# LLM intialized in HuggingFace Pipeline wrapper
llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

# load the knowledge base
loader = CSVLoader('data/test.csv')
docs = loader.load()

# split document into text chunks
# in order to create vector embedding of our data, we need to tokenize our text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs)

# initialize the open-source embedding function, default: text-embedding-ada-002
# this embedding function converts text chunks into tokens
embedding_function = OpenAIEmbeddings()

# load it into ChromaDB
# a database used to store vector embeddings
db = Chroma.from_documents(docs, embedding_function)

# design prompt template
# make the LLM think that they are playing a role by using the template 
template = """You are a customer service chatbot for an online perfume company called Fragrances International.

{context}

Answer the customer's questions only using the source data provided. If you are unsure, say "I don't know, please call our customer support". Use engaging, courteous, and professional language similar to a customer representative.
Keep your answers concise.

Question:

Answer: """

# intiliaze prompt using prompt template via LangChain
prompt = PromptTemplate(template=template, input_variables=["context"])
print(
    prompt.format(
        context = "A customer is on the perfume company website and wants to chat with the website chatbot."
    )
)

# chain to have all components together and query the LLM
chain_type_kwargs = {"prompt": prompt}

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs=chain_type_kwargs,
)

# formatted printing
def print_response(response: str):
    print("\n".join(textwrap.wrap(response, width=80)))

# running chain through LLM with query
query = "What types of perfumes do you sell?"
response = chain.run(query)
print_response(response)