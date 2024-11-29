import streamlit as st
import os
from pymilvus import MilvusClient
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import tempfile
from tqdm import tqdm

# Set environment variables
os.environ["HF_TOKEN"] = "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"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set page configuration
st.set_page_config(page_title="PDF Question Answering", layout="wide")

# Initialize session state
if 'milvus_client' not in st.session_state:
    st.session_state.milvus_client = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "rag_collection"
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None

def initialize_models():
    """Initialize the required models and clients."""
    try:
        # Initialize Milvus Client
        st.session_state.milvus_client = MilvusClient(uri="./hf_milvus_demo.db")
        
        # Create collection if it doesn't exist
        try:
            st.session_state.milvus_client.list_collections()
        except Exception:
            # Create a new collection with the required schema
            st.session_state.milvus_client.create_collection(
                collection_name=st.session_state.collection_name,
                dimension=384  # dimension for bge-small-en-v1.5
            )
        
        # Load embedding model
        st.session_state.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # Initialize LLM client
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        st.session_state.llm_client = InferenceClient(model=repo_id, timeout=120)
        
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        raise e

def emb_text(text):
    """Create embeddings for the given text."""
    return st.session_state.embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def process_pdf(pdf_file):
    """Process the uploaded PDF file and store embeddings in Milvus."""
    # Read PDF
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Create collection if it doesn't exist
    test_embedding = emb_text("This is a test")
    embedding_dim = len(test_embedding)
    
    if st.session_state.milvus_client.has_collection(st.session_state.collection_name):
        st.session_state.milvus_client.drop_collection(st.session_state.collection_name)
    
    st.session_state.milvus_client.create_collection(
        collection_name=st.session_state.collection_name,
        dimension=embedding_dim,
        metric_type="IP",
        consistency_level="Strong",
    )
    
    # Create embeddings and insert into Milvus
    data = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        data.append({
            "id": i,
            "vector": emb_text(chunk),
            "text": chunk
        })
        progress_bar.progress((i + 1) / len(chunks))
    
    insert_res = st.session_state.milvus_client.insert(
        collection_name=st.session_state.collection_name,
        data=data
    )
    return insert_res["insert_count"]

def get_answer(question):
    """Get answer for the given question using RAG pipeline."""
    # Search for relevant chunks
    search_res = st.session_state.milvus_client.search(
        collection_name=st.session_state.collection_name,
        data=[emb_text(question)],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )
    
    # Get retrieved chunks with distances
    retrieved_chunks = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    
    # Create context from retrieved chunks
    context = "\n".join([chunk[0] for chunk in retrieved_chunks])
    
    # Create prompt
    prompt = """
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    If the answer cannot be found in the provided context, respond with EXACTLY this message: "This question is not relevant to the context."
    Do not try to infer or guess information that is not explicitly present in the context.

    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """.format(context=context, question=question)
    
    # Get answer from LLM
    answer = st.session_state.llm_client.text_generation(
        prompt,
        max_new_tokens=1000,
    ).strip()
    
    return answer, retrieved_chunks

# Main app
st.title("📚 PDF Question Answering with RAG")

# Initialize models if not already done
if not st.session_state.milvus_client:
    with st.spinner("Initializing models..."):
        initialize_models()

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Process PDF when uploaded
    with st.spinner("Processing PDF..."):
        insert_count = process_pdf(uploaded_file)
        st.success(f"Successfully processed PDF and created {insert_count} chunks!")

    # Question input
    question = st.text_input("Ask a question about the document:")
    
    if question:
        with st.spinner("Finding answer..."):
            answer, retrieved_chunks = get_answer(question)
            
            # Display answer
            st.markdown("### Answer:")
            st.write(answer)
            
            # Display retrieved chunks with distances
            st.markdown("### Retrieved Context (with similarity scores):")
            for chunk, distance in retrieved_chunks:
                st.info(f"**Similarity Score: {distance:.4f}**\n\n{chunk}")
