from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extract text and split into chunks
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=10,
            chunk_overlap=2,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        st.write("Text Chunks:")
        st.write(chunks)
        
        # Generate embeddings
        embeddings = OpenAIEmbeddings().embed_documents(chunks)
        
        # Handle empty embeddings
        if not embeddings:
            st.error("Error: Empty embeddings. Please check the text content.")
            return
        
        # Create knowledge base
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # Perform similarity search
            docs = knowledge_base.similarity_search(user_question)
            
            # Load question answering chain
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the question answering chain
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
            
            # Display response
            st.write("Answer:")
            st.write(response)
            
            # Debug log: display documents for debugging
            st.write("Documents:")
            st.write(docs)
            
            # Debug log: display embeddings for debugging
            st.write("Embeddings:")
            st.write(embeddings)
        
        # Debug log: display extracted text and chunks for debugging
        st.write("Extracted Text:")
        st.write(text)
        
        st.write("Text Chunks:")
        st.write(chunks)

if __name__ == '__main__':
    main()
