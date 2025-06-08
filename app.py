import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_groq import ChatGroq
from langchain_groq import GroqEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import uuid
import time
import tempfile
import json
from datetime import datetime

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "processed_pdfs" not in st.session_state:
    st.session_state.processed_pdfs = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    processed_files = []
    
    try:
        for pdf in pdf_docs:
            # Save PDF name for tracking
            pdf_name = pdf.name
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf.getvalue())
                temp_path = temp_file.name
            
            try:
                pdf_reader = PdfReader(temp_path)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from each page
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                
                # Add to processed PDFs list with metadata
                processed_files.append({
                    "name": pdf_name,
                    "pages": num_pages,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "size": len(pdf.getvalue()) // 1024  # Size in KB
                })
                
            except Exception as e:
                st.error(f"Error processing {pdf_name}: {str(e)}")
            
            # Clean up temp file
            os.unlink(temp_path)
        
        # Update session state with newly processed PDFs
        st.session_state.processed_pdfs.extend(processed_files)
        
        return text
    except Exception as e:
        st.error(f"Error in PDF processing: {str(e)}")
        return ""

def get_text_chunks(text):
    """Split text into manageable chunks"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error in text chunking: {str(e)}")
        return []

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks"""
    try:
        if not GROQ_API_KEY:
            st.error("GROQ API key is missing. Please add it to your .env file.")
            return None
            
        embeddings = GroqEmbeddings(
            groq_api_key=GROQ_API_KEY,
            model_name="llama2-70b-4096"
        )
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save with session ID to allow multiple users
        vector_store_path = f"faiss_index_{st.session_state.session_id}"
        vector_store.save_local(vector_store_path)
        
        return vector_store_path
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    """Create a conversational chain with memory"""
    try:
        if not GROQ_API_KEY:
            st.error("GROQ API key is missing. Please add it to your .env file.")
            return None
            
        # Create a prompt template
        prompt_template = """
        You are a helpful assistant that answers questions based on the provided PDF documents.
        
        Use the following context to answer the user's question. If the answer is not contained 
        within the context, say "I don't have enough information to answer that based on the 
        uploaded PDFs." Don't try to make up an answer.
        
        Provide detailed, accurate responses and use bullet points or numbered lists when appropriate.
        
        Previous conversation:
        {chat_history}
        
        Context:
        {context}
        
        Question: 
        {question}
        
        Answer:
        """
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        
        # Initialize the LLM
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama2-70b-4096",
            temperature=0.2,
            max_tokens=2048
        )
        
        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question", "chat_history"]
        )
        
        # Create chain
        chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            memory=memory,
            prompt=prompt,
            verbose=True
        )
        
        return chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def user_input(user_question, vector_store_path):
    """Process user question and generate response"""
    try:
        if not GROQ_API_KEY:
            st.error("GROQ API key is missing. Please add it to your .env file.")
            return
            
        # Start timing for response generation
        start_time = time.time()
        
        # Get embeddings
        embeddings = GroqEmbeddings(
            groq_api_key=GROQ_API_KEY,
            model_name="llama2-70b-4096"
        )
        
        # Load vector store
        try:
            new_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            st.error("Please upload and process PDFs before asking questions.")
            return
        
        # Perform similarity search
        docs = new_db.similarity_search_with_score(user_question, k=3)
        
        # Get conversation chain
        chain = get_conversational_chain()
        if not chain:
            return
        
        # Generate response
        response = chain(
            {
                "input_documents": [doc[0] for doc in docs], 
                "question": user_question
            }
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Display retrieved chunks with relevance scores
        with st.expander("View source documents"):
            for i, (doc, score) in enumerate(docs):
                st.markdown(f"**Source {i+1}** (Relevance: {score:.2f})")
                st.markdown(f"```\n{doc.page_content[:500]}...\n```")
        
        # Add to conversation history
        st.session_state.conversation_history.append({
            "question": user_question,
            "answer": response["answer"],
            "sources": [{"content": doc[0].page_content[:300] + "...", "score": float(doc[1])} for doc in docs],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "response_time": f"{response_time:.2f}s"
        })
        
        # Display the answer
        st.markdown("### Answer:")
        st.markdown(response["answer"])
        st.markdown(f"*Response time: {response_time:.2f}s*")
        
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def clear_vector_store():
    """Clear the vector store and reset processed PDFs"""
    try:
        vector_store_path = f"faiss_index_{st.session_state.session_id}"
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
        st.session_state.processed_pdfs = []
        st.success("Vector store and processed PDFs have been cleared.")
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

def export_conversation():
    """Export conversation history as JSON"""
    if st.session_state.conversation_history:
        conversation_json = json.dumps(
            st.session_state.conversation_history, 
            indent=2
        )
        
        # Create download button
        st.download_button(
            label="Download Conversation",
            data=conversation_json,
            file_name=f"pdf_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("No conversation to export.")

def main():
    """Main application function"""
    # Set page config
    st.set_page_config(
        page_title="Chat with PDFs using GROQ",
        page_icon="📚",
        layout="wide"
    )
    
    # Header
    st.header("📚 Chat with Multiple PDFs using GROQ & LangChain")
    
    # Check for API key
    if not GROQ_API_KEY:
        st.warning(
            "GROQ API Key not found. Please add your GROQ API key to the .env file."
            "You can get your API key from https://console.groq.com/keys"
        )
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        # Sidebar for PDF upload and processing
        st.subheader("📁 Document Management")
        
        # PDF uploader
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type="pdf"
        )
        
        # Process button
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Process PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            # Create vector store
                            vector_store_path = get_vector_store(text_chunks)
                            if vector_store_path:
                                st.session_state.vector_store_path = vector_store_path
                                st.success(f"✅ {len(pdf_docs)} PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")
        
        # Display processed PDFs
        if st.session_state.processed_pdfs:
            st.subheader("📋 Processed PDFs")
            for i, pdf in enumerate(st.session_state.processed_pdfs):
                st.markdown(f"**{i+1}. {pdf['name']}**")
                st.markdown(f"Pages: {pdf['pages']} | Size: {pdf['size']} KB | Added: {pdf['timestamp']}")
            
            # Clear button
            if st.button("Clear All PDFs"):
                clear_vector_store()
        
        # Export conversation
        if st.session_state.conversation_history:
            st.subheader("💾 Export")
            export_conversation()
    
    with col1:
        # Main chat interface
        st.subheader("💬 Ask about your PDFs")
        
        # Display conversation history
        if st.session_state.conversation_history:
            for i, exchange in enumerate(st.session_state.conversation_history):
                # User question
                st.markdown(f"**You:** {exchange['question']}")
                # AI answer
                st.markdown(f"**AI:** {exchange['answer']}")
                st.markdown("---")
        
        # Question input
        user_question = st.text_input("Ask a question about your PDFs:")
        
        # Process question
        if user_question:
            if hasattr(st.session_state, 'vector_store_path'):
                user_input(user_question, st.session_state.vector_store_path)
            else:
                st.warning("Please upload and process PDFs before asking questions.")

if __name__ == "__main__":
    main()
