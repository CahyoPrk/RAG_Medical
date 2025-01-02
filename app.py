import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import tempfile
import cv2
import pysqlite3 as sqlite3

# Configurations
st.set_page_config(page_title="RAG for Medical Record Extraction", layout="wide")

# Sidebar - API Key Input
st.sidebar.header("Configuration")
google_api_key = st.sidebar.text_input("Google API Key", type="password")

def main():
    st.title("Extract Information from Medical Records (Scanned PDF)")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload a Scanned PDF file", type="pdf")

    if uploaded_file:
        if google_api_key:
            genai.configure(api_key=google_api_key)

            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            # Show button to process the file
            st.info("Click the button below to extract information from the uploaded PDF.")

            # Button to start extraction after file upload
            if st.button("Ekstrak Informasi"):
                # Load and Process PDF
                st.subheader("Processing PDF... Please wait a moment.")
                pdf_loader = PyPDFLoader(temp_file_path, extract_images=True)
                pages = pdf_loader.load_and_split()
                context = "\n\n".join(str(p.page_content) for p in pages)

                # Text Splitting
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                texts = text_splitter.split_text(context)

                # Embedding Model
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", google_api_key=google_api_key
                )

                # Create Vector Index
                vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

                # LLM Model (Gemini for structured extraction)
                model = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=google_api_key,
                    temperature=0.2,
                    convert_system_message_to_human=True
                )

                # RAG Chain for Information Extraction
                qa_chain = RetrievalQA.from_chain_type(
                    model, retriever=vector_index, return_source_documents=True
                )

                # The input query for the extraction task
                query = """
                Extract the following information from the medical record:
                1. Nama Pasien
                2. Tanggal Lahir
                3. Nama Dokter
                4. Diagnosa
                5. Total Biaya
                Please ensure the data is structured and clearly separated.
                """

                with st.spinner("Generating structured data..."):
                    result = qa_chain({"query": query})
                    structured_data = result["result"]

                    # Display Results
                    st.subheader("Extracted Information")
                    st.write(structured_data)

                    st.subheader("Source Documents")
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)
        else:
            st.error("Please provide your Google API Key in the sidebar.")
    else:
        st.info("Please upload a scanned PDF file to proceed.")


if __name__ == "__main__":
    main()
