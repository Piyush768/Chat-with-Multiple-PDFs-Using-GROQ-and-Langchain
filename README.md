# **📄 Chat with PDF using GROQ & LangChain**
A **Streamlit-based chatbot** that allows users to upload PDFs and interact with their content using **GROQ AI**, **LangChain**, and **FAISS**.

---

## **🚀 Features**
- **Upload PDFs**: Extracts text from PDF files.
- **Question Answering**: Asks questions based on the PDF content.
- **AI-Powered Responses**: Uses **Google Generative AI** for detailed answers.
- **FAISS Vector Search**: Fast text retrieval from processed chunks.
- **User-Friendly UI**: Interactive web app built with Streamlit.

---

## **🔧 Installation**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-github-username/chat-with-pdf-groq.git
   cd chat-with-pdf-groq
   ```
2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
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

---

## **🛠️ Usage**
1. **Run the App**:
   ```bash
   streamlit run app.py
   ```
2. **Upload PDF Files**:
   - Use the sidebar to upload PDFs.
   - Click "Submit & Process" to process the files.
3. **Ask Questions**:
   - Enter a question in the input box to get AI-powered responses.

---

## **📂 Project Structure**
- `app.py`: Main application file.
- `requirements.txt`: Dependencies.
- `.env`: API key storage (not included in Git).

---

## **📜 License**
Licensed under the **MIT License**. Feel free to use and modify the code.

---

## **🌟 Star This Repo**
If you find this project helpful, don’t forget to give it a ⭐ on GitHub!

---

This concise `README.md` provides the essential details while remaining user-friendly. Let me know if you need further tweaks! 🚀
