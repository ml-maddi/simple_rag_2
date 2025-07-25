# main.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- CONFIGURATION & CONSTANTS ---
load_dotenv()

# Load Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it as an environment variable.")

PDF_FILE_PATH = "HSC26_Bangla_1st_paper.pdf"
VECTOR_DB_PATH = "chroma_db_multilingual"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" # A robust and efficient model
LLM_MODEL_NAME = "gemini-2.5-flash"
PROMPT_TEMPLATE = """You are a helpful AI assistant for answering questions about a given document.
You are given a question and a set of document chunks as context.
You must STRICTLY FOLLOW these rules:
1. YOUR FINAL ANSWER MUST BE IN THE SAME LANGUAGE AS THE USER'S QUESTION.
2. IF THE USER'S QUESTION IS IN English, YOUR ANSWER MUST BE IN English.
3. IF THE USER'S QUESTION IS IN Bangla, YOUR ANSWER MUST BE IN Bangla.
4. If the information to answer the question is not in the context, you MUST respond with one of the following sentences, matching the language of the question:
   - For English questions: "Sorry, I am unable to answer this question from the provided document."
   - For Bangla questions: "দুঃখিত, আপনার প্রশ্নটির উত্তর আমার জানা নেই।"
5. Do not make up answers. Your response must be grounded in the provided context.

Context:
{context}

Question: {question}
Answer:"""

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="Multilingual RAG Chatbot API",
    description="An API for a RAG chatbot that answers questions about a document in English and Bengali.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- GLOBAL VARIABLES & STATE ---
# We will store the chain in the app state to initialize it once on startup
app.state.chain = None

# --- HELPER FUNCTIONS ---

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def detect_language(text):
    bangla_count = sum(0x0980 <= ord(char) <= 0x09FF for char in text)
    english_count = sum(0x0041 <= ord(char) <= 0x007A or 0x0030 <= ord(char) <= 0x0039 for char in text)
    
    if bangla_count > english_count:
        return "Bangla"
    elif english_count > bangla_count:
        return "English"
    else:
        return "Mixed or Unknown"

def load_and_embed_pdf():
    """
    Loads the PDF, splits it into chunks, creates embeddings,
    and stores them in a Chroma vector database.
    """
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing vector database...")

        return Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=hf
        )

    if not os.path.exists(PDF_FILE_PATH):
        print(f"PDF file not found at {PDF_FILE_PATH}. Creating a dummy file.")
        with open(PDF_FILE_PATH, "w") as f:
            f.write("This is a placeholder PDF. Please replace it with your actual document.")

    print(f"Loading and processing '{PDF_FILE_PATH}'...")
    loader = PyMuPDFLoader(PDF_FILE_PATH)
    documents = loader.load()

    for doc in documents:
        doc.page_content = doc.page_content.replace('\n', ' ').strip()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not split the document into chunks. Check the PDF content.")

    print("Creating vector embeddings... This is a one-time process.")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=hf,
        persist_directory=VECTOR_DB_PATH
    )
    print("Vector database created and persisted successfully!")
    return vector_store

def initialize_llm_and_chain(vector_store):
    """
    Initializes the LLM and the conversational retrieval chain.
    """
    if vector_store is None:
        return None

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    qa_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return chain


# --- STARTUP EVENT ---

@app.on_event("startup")
def on_startup():
    """
    This function runs when the FastAPI application starts.
    It initializes the RAG chain.
    """
    print("Application startup: Initializing RAG chain...")
    try:
        vector_store = load_and_embed_pdf()
        app.state.chain = initialize_llm_and_chain(vector_store)
        if app.state.chain:
            print("RAG chain initialized successfully.")
        else:
            print("Error: RAG chain could not be initialized.")
    except Exception as e:
        print(f"An error occurred during startup initialization: {e}")
        app.state.chain = None

# --- API & FRONTEND ENDPOINTS ---

@app.get("/", response_class=FileResponse, tags=["Frontend"])
async def read_index():
    """Serves the frontend HTML file."""
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found")
    return "index.html"

@app.get("/api/status", tags=["Status"])
def get_status():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the RAG Chatbot API!"}

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    source_documents: list

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_document(request: ChatRequest):
    """
    The main endpoint for chatting with the document.
    Receives a query and returns the answer and source documents.
    """
    if not app.state.chain:
        raise HTTPException(
            status_code=503,
            detail="The RAG chain is not initialized. Please check the server logs."
        )
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        language = detect_language(request.query)
        print(f"language: {language}")
       # 2. Create a new, formatted question that includes the language instruction
        # This is the key change to make the chain work correctly.
        formatted_question = f"Please answer in {language}. User question: '{request.query}'"

        # 3. Call the chain with the standard input format it expects
        result = app.state.chain({"question": formatted_question})
        
        source_docs_formatted = []
        for doc in result.get('source_documents', []):
            source_docs_formatted.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        return ChatResponse(
            answer=result.get('answer', 'No answer found.'),
            # source_documents=source_docs_formatted
        )
    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)