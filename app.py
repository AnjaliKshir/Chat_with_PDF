import streamlit as st
from PyPDF2 import PdfReader
from langchain import RecursiveCharacterTextSplitter
import os
from langchian_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchian_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#function to extract text from each page of the pdfs
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

#splitting the extracted text in smaller chunks
def create_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)

    return chunks

#generate vector embeddings of the generated text_chunks and save locally
def generate_vector_embeddings(text_chunks):
    vector_embedding_technique = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_embeddings = FAISS.from_texts(text_chunks, embedding = vector_embedding_technique)  
    vector_embeddings.save_local("faiss_index")



    