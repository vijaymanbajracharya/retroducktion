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

# Set up HuggingFace Pipeline with Llama-2-7b-chat-hf model
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

if __name__ == '__main__':
    pass