# **📚 Chat with Multiple PDFs using GROQ & LangChain**

A **Streamlit-based chatbot** that allows users to upload multiple PDFs and interact with their content using **GROQ AI**, **LangChain**, and **FAISS** vector database.

![App Screenshot](https://via.placeholder.com/800x400?text=PDF+Chat+App)

---

## **🚀 Features**

- **Upload Multiple PDFs**: Extract text from multiple PDF files at once
- **Conversation Memory**: Maintains context across multiple questions
- **Advanced Question Answering**: Ask questions based on the PDF content
- **GROQ AI Integration**: Uses GROQ's powerful LLMs for detailed answers
- **FAISS Vector Search**: Fast text retrieval from processed chunks
- **Source References**: View the exact sources used to generate answers
- **Conversation Export**: Save your chat history as JSON
- **Document Management**: See all processed PDFs and clear them when needed
- **Error Handling**: Robust error handling for a smooth user experience
- **Responsive UI**: Clean, interactive web app built with Streamlit

---

## **🔧 Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-github-username/chat-with-multiple-pdfs-groq.git
   cd chat-with-multiple-pdfs-groq
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # OR
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up API Key**:
   - Create a `.env` file and add your **GROQ API Key**:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - You can get your GROQ API key from [https://console.groq.com/keys](https://console.groq.com/keys)

---

## **🛠️ Usage**

1. **Run the App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF Files**:
   - Use the document management panel to upload one or more PDFs
   - Click "Process PDFs" to extract text and create embeddings

3. **Ask Questions**:
   - Enter questions in the input box to get AI-powered responses
   - View source documents to see where the information came from
   - Continue the conversation with follow-up questions

4. **Manage Documents**:
   - View all processed PDFs with metadata
   - Clear all PDFs when you want to start fresh

5. **Export Conversation**:
   - Download your conversation history as a JSON file

---

## **📂 Project Structure**

- `app.py`: Main application file with all functionality
- `requirements.txt`: Project dependencies
- `.env`: API key storage (not included in Git)

---

## **🔍 How It Works**

1. **PDF Processing Pipeline**:
   - Extract text from PDFs
   - Split text into manageable chunks
   - Create vector embeddings using GROQ
   - Store embeddings in FAISS vector database

2. **Question Answering System**:
   - Retrieve relevant chunks based on question similarity
   - Provide context to GROQ LLM
   - Generate detailed, contextual answers
   - Maintain conversation history for follow-up questions

---

## **🧩 Advanced Usage**

- **Adjust Chunk Size**: Modify the `chunk_size` and `chunk_overlap` parameters in the code to optimize for your specific PDFs
- **Change LLM Model**: Update the model name in the `ChatGroq` initialization to use different GROQ models
- **Customize Prompts**: Edit the prompt template to change how the AI responds

---

## **📝 License**

Licensed under the **MIT License**. Feel free to use and modify the code for your own projects.

---

## **🙏 Acknowledgements**

- [Streamlit](https://streamlit.io/) for the web app framework
- [LangChain](https://langchain.com/) for the RAG pipeline
- [GROQ](https://groq.com/) for the LLM API
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage
