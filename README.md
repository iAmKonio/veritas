# veritas

'veritas' is Latin for "truth" - it implies this application finds true answers from documents. It will tell you if it doesn't know something.

---
## 📄 RAG Chatbot for Company Documents

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** designed to answer questions based on your company's internal `.txt` and `.pdf` documents. It leverages the power of **LangChain** for document processing and conversational capabilities, **HuggingFace Embeddings** for semantic search, and **Google's Gemini 1.5 Flash** model for generating human-like responses. The chatbot is presented through a user-friendly web interface built with **Gradio**.

### ✨ Features

* **Document Loading:** Easily ingest `.txt` and `.pdf` files from a specified local directory (`./docs`).
* **Intelligent Retrieval:** Utilizes **ChromaDB** as a vector store to efficiently retrieve relevant document chunks based on your queries.
* **Conversational AI:** Powered by **Google Gemini 1.5 Flash** for understanding questions and generating coherent answers.
* **Contextual Understanding:** Maintains conversation history to provide more accurate and contextually relevant responses.
* **User-Friendly Interface:** An interactive chat interface built with **Gradio** for seamless interaction.
* **Python-based:** Developed entirely in Python, making it easy to understand and extend.

### ⚙️ How it Works

1.  **Document Loading:** The application scans the `./docs` folder for `.txt` and `.pdf` files and loads their content.
2.  **Text Splitting:** Loaded documents are split into smaller, manageable chunks to optimize retrieval and processing.
3.  **Embedding:** Each text chunk is converted into numerical vectors (embeddings) using **HuggingFace's `all-MiniLM-L6-v2`** model. These embeddings capture the semantic meaning of the text.
4.  **Vector Storage:** The embeddings are stored in a **ChromaDB** vector store, enabling fast and efficient similarity searches.
5.  **Question Answering (RAG):**
    * When a user asks a question, the question is also embedded.
    * The system then retrieves the most semantically similar document chunks from ChromaDB.
    * These retrieved chunks, along with the user's question and conversation history, are fed to the **Gemini 1.5 Flash LLM**.
    * The LLM generates a comprehensive answer based on the provided context.

### 🚀 Setup and Installation

To get this chatbot up and running, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone [your-repository-url]
    cd [your-repository-name]
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    (If you don't have a `requirements.txt`, you'll need to create one. Based on your code, it should include: `python-dotenv`, `gradio`, `langchain`, `langchain-community`, `langchain-google-genai`, `pypdf`, `sentence-transformers`, `chromadb`)

4.  **Set Up Google API Key:**
    * Obtain a **Google Gemini API key** from the [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create a `.env` file in the root directory of your project.
    * Add your API key to the `.env` file like this:
        ```
        GOOGLE_API_KEY="your_google_gemini_api_key_here"
        ```

5.  **Place Your Documents:**
    * There is a folder named `docs` in the root directory of your project.
    * Place your `.txt` and `.pdf` company documents inside this `docs` folder.

6.  **Run the Application:**

    ```bash
    python veritas.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file, e.g., `chatbot_app.py`)

    The application will start, and you'll see a local URL (e.g., `http://0.0.0.0:7860`) in your terminal. Open this URL in your web browser to access the chatbot. If `share=True` is enabled in `demo.launch()`, you'll also get a public Gradio share link.

### 💡 Usage

Once the Gradio interface is loaded in your browser:

1.  Type your question into the "Your Question" textbox.
2.  Click the "Submit" button or press Enter.
3.  The chatbot will process your request and display the answer in the chat window.
4.  Use the "Clear" button to clear the chat history and the question box.

---
