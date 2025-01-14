import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
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
    text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    chunks = text_splitter.split_text(text)

    return chunks

#generate vector embeddings of the generated text_chunks and save locally
def generate_vector_embeddings(text_chunks):
    vector_embedding_technique = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_embeddings = FAISS.from_texts(text_chunks, embedding = vector_embedding_technique)  
    vector_embeddings.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the questiona as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context say "The answer is not available in the provided context.", do not provide wrong answers \n\n
    Context: \n {context}? \n
    Question: \n {question} \n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) 
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)

    return chain

# Function to show reply box when a reply is generated
def display_reply(reply):
    if reply:
        st.markdown(f'<div class="reply-box">{reply}</div>', unsafe_allow_html=True)
#     else:
#         # Hide the reply box if no reply
#         st.markdown('<div class="reply-box hidden"></div>', unsafe_allow_html=True)

    
def user_input(user_question):
    #initialize the embedding model
    vector_embedding_technique = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    #load the FAISS index - which contains the vector embeddings of the PDF text chunks
    new_db = FAISS.load_local("faiss_index", vector_embedding_technique, allow_dangerous_deserialization=True)

    # Convert user question to embeddings manually
    # question_embedding = vector_embedding_technique.embed(user_question)

    # Use the embedding for similarity search
    docs = new_db.similarity_search(user_question)
    # docs = new_db.similarity_search(user_question)

    # Load the conversational chain
    chain = get_conversational_chain()

    #Generate a response using the chain
    response = chain({"input_documents" : docs, "question" : user_question}, return_only_outputs = True)
    print(response)

    #Display the response in the Streamlit UI
    # st.write("Reply: ", response["output_text"])
    # display_reply(response["output_text"])
    return response["output_text"]

# Main function to create the Streamlit app with aesthetic frontend
def main():
    # Set the title of the web app
    st.set_page_config(page_title="Chat with PDF", page_icon="📚")

    # Create a header with a custom style
    st.markdown("""
    <h1 style='text-align: center; color: #FFB6C1;'>Chat with Your PDFs</h1>
    <p style='text-align: center; color: #8B8B8B;'>Ask questions based on the content of your uploaded PDF documents.</p>
        
    """, unsafe_allow_html=True)

    # Sidebar for uploading PDF documents and processing
    st.sidebar.title("Upload PDFs")

    # Browse files button
    uploaded_files = st.sidebar.file_uploader("Browse Files", type=["pdf"], accept_multiple_files=True)

    # Button for starting the processing when the user clicks "Submit and Process"
    if st.sidebar.button("Submit and Process"):
        if uploaded_files:
            pdf_text = extract_pdf_text(uploaded_files)
            if pdf_text.strip():
                text_chunks = create_text_chunks(pdf_text)
                generate_vector_embeddings(text_chunks)
                st.sidebar.success("PDFs uploaded successfully!")
            else: 
                st.sidebar.warning("No text found in the uploaded PDFs.")
        else:
            st.sidebar.warning("Please upload PDF files before processing!")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.markdown(f'<div class="chat-box {message["role"]}">{message["content"]}</div>', unsafe_allow_html=True)

    # Text input for the user question
    user_question = st.text_input("", placeholder="Type your question here...", key="question", label_visibility="hidden")

    # Trigger question processing when the user presses Enter
    if user_question:
        # Add user input to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        response = user_input(user_question)
        display_reply(response)
        
        # Add system reply to chat history
        st.session_state.messages.append({"role": "system", "content": response})
    

    # Add custom interactivity on button hover (glowing effect)
    # if st.button("Get Answer", key="submit"):
    #     if user_question:
    #         st.session_state.messages.append({"role": "user", "content": user_question})
    #         user_input(user_question)
    #         st.session_state.messages.append({"role": "system", "content": "Response will appear here."})
    #     else:
    #         st.warning("Please enter a question to get a response.")

    # Apply custom CSS to enhance the look
    st.markdown("""
    <style>
        
        .chat-box {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 10px;
            max-width: 100%;
            font-family: Arial, sans-serif;
        }

        .chat-box.user {
            background-color: #FFB6C1;
            color: white;
            text-align: left;
            display: flex;
            align-items:flex-end;
            width:40%;
        }

        .chat-box.system {
            background-color: #F5EFFF;
            color: #333;
            text-align: left;
        }
        


        .stTextInput{
            outline: none;
            z-index: 1000;
            padding: 0px;
            border-radius: 10px;
            border: 2px solid #FFB6C1;
            transition: border-color 0.3s ease;
            position: fixed;
            bottom: 8%;
            max-width: 100vw;
            display:flex;
            align-items: center;
            margin: 10px;
        }
                
        .stTextInput:hover {
            outline:none;
            border-color: #ff9fcf;
            transition: 0.6s;
        }  
                     
        .stTextInput :focus {
            outline: none;
            border-color: #FF69B4;
            transition: 0.6s;
                
        }
        .user-message {
            background-color: #FFB6C1;
            border-radius: 12px;
            padding: 10px;
            margin: 5px;
            font-family: Arial, sans-serif;
            color: #fff;
        }
        .system-message {
            background-color: #8B8B8B;
            border-radius: 12px;
            padding: 10px;
            margin: 5px;
            font-family: Arial, sans-serif;
            color: #fff;
        }
                
        
        .reply-box {
        display: box;
        /*background-color: #E1EACD;
        border: 2px solid #BAD8B6;*/
        background-color: #F5EFFF;
        border: 2px solid #A294F9;
        padding: 10px;
        border-radius: 10px;
        font-family: Arial, sans-serif;
        color: #333;
        max-width: 100%;
        margin-bottom: 20px;
        }      
        .reply-box:hover{
        
        }         
        .hidden{
            display:none;    
        }
                
        
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()