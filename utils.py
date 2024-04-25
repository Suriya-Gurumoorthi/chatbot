from sentence_transformers import SentenceTransformer
import pinecone
import openai
import os
from langchain_pinecone.vectorstores import Pinecone
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
openai_api_key = "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"

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings

PINECONE_API_KEY = "eeaf25f5-f2f3-477b-984d-13f6d71fc448"
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
index = pinecone.Index(name='langchain-vectors',host="https://langchain-vectors-k0at4h7.svc.aped-4627-b74a.pinecone.io",api_key="eeaf25f5-f2f3-477b-984d-13f6d71fc448")

def find_match(input):
    input_em=embeddings.embed_query(input)
    result = index.query(vector = input_em, top_k=10, includeMetadata=True)
    if 'matches' in result:
        if len(result['matches']) >= 2:
            match1_metadata = result['matches'][0].get('metadata', {}).get('text', 'Metadata not found')
            match2_metadata = result['matches'][1].get('metadata', {}).get('text', 'Metadata not found')
            return match1_metadata + "\n" + match2_metadata
        else:
            print("Less than 2 matches found.")
    else:
        print("No matches found.")

    return "No matches found."

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
