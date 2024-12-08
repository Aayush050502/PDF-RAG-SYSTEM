import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    return vector_store

def save_vectorstore(vector_store):
    # Ensure the directory exists
    os.makedirs('vectorstore', exist_ok=True)
    
    try:
        # Save index and metadata separately
        vector_store.save_local('vectorstore')
        
        # Create a metadata file to track the vector store
        metadata = {
            'num_docs': vector_store.index.ntotal,
            'dimension': vector_store.index.d
        }
        
        with open('vectorstore/metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        st.success("Vector store saved successfully!")
    except Exception as e:
        st.error(f"Error saving vector store: {e}")

def load_vectorstore():
    try:
        # Check if vector store files exist
        if not os.path.exists('vectorstore/index.faiss'):
            st.warning("No existing vector store found.")
            return None
        
        # Load embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load vector store
        vector_store = FAISS.load_local(
            'vectorstore', 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Optional: Load and display metadata
        try:
            with open('vectorstore/metadata.json', 'r') as f:
                metadata = json.load(f)
                st.info(f"Loaded vector store with {metadata['num_docs']} documents")
        except Exception:
            pass
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        # Load vector store
        vector_store = load_vectorstore()
        
        if vector_store is None:
            st.error("Please upload and process PDFs first.")
            return

        # Perform similarity search
        docs = vector_store.similarity_search(user_question)

        # Get conversational chain
        chain = get_conversational_chain()
        
        # Generate response
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )

        # Display response
        st.write("Reply: ", response["output_text"])

    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multiple PDF using Cerebral Zip💁")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Process user question if vector store exists
    if user_question:
        user_input(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks)
                    
                    # Save vector store
                    save_vectorstore(vector_store)
                    
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

if __name__ == "__main__":
    main()