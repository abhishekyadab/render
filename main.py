import os
from dotenv import load_dotenv
import glob
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from google import genai
from tqdm import tqdm
from flask import Flask, request, jsonify
import subprocess

# 1. Load Gemini API key from .env or environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# 2. Create Gemini client (new SDK)
client = genai.Client(api_key=api_key)

# 3. Ensure Godspeed docs are present (clone if not)
if not os.path.exists("gs-documentation"):
    print("Cloning Godspeed documentation repo...")
    subprocess.run(["git", "clone", "https://github.com/godspeedsystems/gs-documentation.git"])

repo_path = "gs-documentation"

# 4. Read all relevant files from the repo (including .yaml/.yml)
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
                print(f"Error reading {file_path}: {e}")
    return documents

print("Reading repository files...")
documents = read_repository_files(repo_path)
print(f"Found {len(documents)} files")

# 5. Split documents into chunks
def split_documents(documents):
    chunks = []
    md_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    code_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for doc in tqdm(documents):
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

print("Splitting documents into chunks...")
chunks = split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 6. Embed chunks using Gemini
def embed(text):
    result = client.models.embed_content(
        model="models/embedding-001",
        contents=text
    )
    return result.embeddings[0].values

print("Creating embeddings...")
embeddings = [embed(chunk["content"]) for chunk in tqdm(chunks)]

# 7. Store embeddings in ChromaDB
print("Storing embeddings in ChromaDB...")
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.create_collection("godspeed_docs")
for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings))):
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk["content"]],
        metadatas=[chunk["metadata"]]
    )

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

def rag_query(query):
    # Enhance query for auth-related questions
    enhanced_query = enhance_jwt_auth_query(query)
    
    # Embed the enhanced query
    query_embedding = embed(enhanced_query)
    
    # Retrieve more relevant chunks (increased from 3 to 7)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=7,  # Increased to get more context and examples
        include=["documents", "metadatas"]
    )
    
    # Construct context from retrieved documents
    context = ""
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        context += f"\nFrom file: {source}\n{doc}\n"
    
    # Enhanced prompt that explicitly requests the full event source config block
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
    generation = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    
    return generation.text

# 9. Flask API for deployment
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query', '')
    answer = rag_query(user_query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    print("Godspeed Documentation RAG Agent Ready!")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
