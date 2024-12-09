import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import os
from streamlit_chat import message

load_dotenv()

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Function to perform RAG (provided by you)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),)

# Connect to your Pinecone index
pinecone_index = pc.Index("stocks")
def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace="https://github.com/CoderAgent/SecureAgent"
    )

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as needed to improve the response quality
    system_prompt = f"""You have 20+ years of experience of stock market and you know all ins and out. 
    You are an expert at providing answers about stocks. Please answer my question provided. 
    Your advice make people Millionaire. You analyze my question and answer it very thoughtfully.  
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

st.title("Financial Analysis & Automation")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you, future millionaire?"}
    ]

# Display existing messages with unique keys
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user query
if user_input := st.chat_input("Enter your question:", key="user_input"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the user query with perform_rag
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

        # Call perform_rag to get the assistant's response
        response = perform_rag(user_input)

        # Display the assistant's response
        response_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
