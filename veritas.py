import os
import warnings
import gradio as gr

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Load Environment Variables ---
load_dotenv()  # Loads GOOGLE_API_KEY from .env file

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")

# --- Load Local Documents (.txt and .pdf) ---
def load_documents_from_folder(folder_path):
    documents = []
    for filepath in Path(folder_path).glob("*"):
        if filepath.suffix.lower() == ".txt":
            loader = TextLoader(str(filepath))
        elif filepath.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(filepath))
        else:
            continue
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {filepath.name}: {e}")
    return documents

folder_path = "./docs"
print(f"Loading documents from: {folder_path}")
documents = load_documents_from_folder(folder_path)
print(f"✅ Loaded {len(documents)} documents.")

# --- Split, Embed, and Store ---
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"✅ Split into {len(texts)} chunks.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Chroma.from_documents(texts, embeddings)
print("✅ Documents embedded and stored in ChromaDB.")

# --- LLM & Memory Setup ---
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=gemini_llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    memory=memory,
    return_source_documents=False
)

# --- Chat Function ---
def chat_with_rag(message, history_list):
    result = qa_chain.invoke({"question": message})
    bot_response = result["answer"]
    history_list.append([message, bot_response])
    return history_list, ""

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Company Document Chatbot")
    gr.Markdown("Ask any question based on the loaded `.txt` and `.pdf` files in the `docs/` folder.")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Question", placeholder="Ask something about company policies...", lines=2)

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.ClearButton([msg, chatbot])

    msg.submit(chat_with_rag, inputs=[msg, chatbot], outputs=[chatbot, msg])
    submit_btn.click(chat_with_rag, inputs=[msg, chatbot], outputs=[chatbot, msg])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)