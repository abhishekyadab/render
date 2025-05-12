import os
import streamlit as st
from dotenv import load_dotenv
import glob
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from google import genai
from tqdm import tqdm
import subprocess
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="Godspeed Documentation RAG Agent", layout="wide")

# 1. Load Gemini API key from environment (set in Streamlit Secrets)
api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found! Please add it to your .streamlit/secrets.toml file.")
    st.stop()

# 2. Create Gemini client
client = genai.Client(api_key=api_key)

# Simple in-memory vector store (replacement for ChromaDB)
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
    
    def add(self, ids, embeddings, documents, metadatas):
        for doc, emb, meta in zip(documents, embeddings, metadatas):
            self.documents.append(doc)
            self.embeddings.append(emb)
            self.metadatas.append(meta)
    
    def query(self, query_embeddings, n_results=5, include=None):
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]]}
            
        similarities = cosine_similarity([query_embeddings], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        
        results = {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]]
        }
        return results

# 3. Clone Godspeed docs repo if not present
@st.cache_resource
def clone_repo():
    if not os.path.exists("gs-documentation"):
        with st.spinner("Cloning Godspeed documentation repo..."):
            subprocess.run(["git", "clone", "https://github.com/godspeedsystems/gs-documentation.git"])
    return "gs-documentation"

# 4. Read all relevant files from the repo
@st.cache_data
def read_repository_files(repo_path):
    documents = []
    extensions = ['.md', '.mdx', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml']
    for ext in extensions:
        for file_path in glob.glob(f"{repo_path}/**/*{ext}", recursive=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    relative_path = os.path.relpath(file_path, repo_path)
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": relative_path,
                            "type": ext[1:]
                        }
                    })
            except Exception as e:
                st.warning(f"Error reading {file_path}: {e}")
    return documents

# 5. Split documents into chunks
@st.cache_data
def split_documents(documents):
    chunks = []
    md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    code_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        splitter = md_splitter if metadata["type"] in ["md", "mdx"] else code_splitter
        doc_chunks = splitter.split_text(content)
        for chunk in doc_chunks:
            chunks.append({
                "content": chunk,
                "metadata": metadata
            })
    return chunks

# 6. Embed chunks using Gemini
def embed(text):
    try:
        result = client.models.embed_content(
            model="models/embedding-001",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        st.error(f"Embedding error: {e}")
        # Return a zero vector as fallback
        return [0.0] * 768

# 7. Initialize the RAG system
@st.cache_resource
def initialize_rag_system():
    vector_store = SimpleVectorStore()
    repo_path = clone_repo()
    
    with st.spinner("Reading repository files..."):
        documents = read_repository_files(repo_path)
    st.success(f"Found {len(documents)} files")
    
    with st.spinner("Splitting documents into chunks..."):
        chunks = split_documents(documents)
    st.success(f"Created {len(chunks)} chunks")
    
    with st.spinner("Creating embeddings... This may take a while"):
        embeddings = []
        for i, chunk in enumerate(chunks):
            if i % 20 == 0:  # Update progress
                st.write(f"Processing chunk {i+1}/{len(chunks)}")
            emb = embed(chunk["content"])
            embeddings.append(emb)
    
    with st.spinner("Storing embeddings..."):
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_store.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[chunk["content"]],
                metadatas=[chunk["metadata"]]
            )
    
    return vector_store

# 8. Enhanced RAG query function
def enhance_jwt_auth_query(query):
    """Add specific file references for authentication queries"""
    if any(term in query.lower() for term in ["jwt", "authentication", "auth", "oauth"]):
        return (query +
                " Provide the full event source configuration block for JWT authentication as shown in the official Godspeed documentation, "
                "including type, port, and the authn section for JWT. "
                "Highlight that JWT authentication is enabled for all endpoints by default (zero-trust). "
                "Show the code block as it appears in http.yaml and explain the purpose of each line.")
    return query

def rag_query(vector_store, query):
    # Enhance query for auth-related questions
    enhanced_query = enhance_jwt_auth_query(query)
    
    # Embed the enhanced query
    query_embedding = embed(enhanced_query)
    
    # Retrieve more relevant chunks
    results = vector_store.query(
        query_embeddings=query_embedding,
        n_results=7,
        include=["documents", "metadatas"]
    )
    
    # Construct context from retrieved documents
    context = ""
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"] if results["metadatas"][0] else "Unknown"
        context += f"\nFrom file: {source}\n{doc}\n"
    
    # Enhanced prompt
    prompt = f"""
You are an expert on the Godspeed framework. Using ONLY the context below, answer the question as follows:

- If the answer involves JWT authentication, ALWAYS include the complete event source configuration block (including `type`, `port`, and the `authn` section for JWT) exactly as shown in the official Godspeed documentation.
- Format the configuration as a markdown code block with YAML syntax.
- Clearly specify the file path (e.g., `src/eventsources/http.yaml`) and explain the purpose of each line.
- State that JWT authentication is enabled for all endpoints by default (zero-trust policy).
- If you don't know the answer based on the context, say so.

Context:
{context}

Question: {query}
"""
    
    # Generate response
    try:
        generation = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return generation.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main Streamlit app
st.title("🚀 Godspeed Documentation RAG Agent")
st.markdown("Ask questions about the Godspeed framework and get answers from the documentation.")

# Initialize RAG system
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses Retrieval-Augmented Generation (RAG) to answer questions about the Godspeed framework.
    
    It processes the official Godspeed documentation and uses Google Gemini to generate answers.
    """)
    if st.button("Initialize/Reset RAG System"):
        st.cache_resource.clear()
        st.experimental_rerun()

# Get or initialize vector store
try:
    vector_store = initialize_rag_system()
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask a question about Godspeed..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation and generating answer..."):
                response = rag_query(vector_store, prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
except Exception as e:
    st.error(f"Error initializing RAG system: {str(e)}")
    st.info("Please try refreshing the page or check the logs for more details.")
