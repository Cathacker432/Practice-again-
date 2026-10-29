import anthropic
import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file

MAX_DISTANCE = 1.3  # chunks further than this are not related to the question
N_RESULTS = 3       # how many chunks to send to Claude

# same database, collection and embedding model as ingest.py
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("sops")
embedder = embedding_functions.DefaultEmbeddingFunction()
claude = anthropic.Anthropic()

SYSTEM_PROMPT = """You are an emergency SOP assistant.
Answer ONLY using the protocol text provided in the context.
If the answer is not in the context, say: "This is not covered in the protocols."
Keep answers short and give steps as a numbered list.
In a real emergency, remind the user to call emergency services (997 in the UAE)."""


# Step 1: page layout
st.title("🚨 Emergency SOP Assistant")
st.write("Ask a question about the fire and flood protocols.")

question = st.text_input("Your question")

if st.button("Ask") and question:

    # Step 2: convert the question into a vector (same all-MiniLM-L6-v2 model as ingest.py)
    question_vector = embedder([question])

    # find the chunks whose vectors are most similar to the question vector
    results = collection.query(query_embeddings=question_vector, n_results=N_RESULTS)
    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    # keep only the chunks that are close enough to the question
    relevant = [(c, s) for c, s, d in zip(chunks, sources, distances) if d < MAX_DISTANCE]

    if not relevant:
        st.warning("This is not covered in the protocols.")
    else:
        # Step 3: send the question and the chunks to Claude
        context = "\n\n---\n\n".join(c for c, s in relevant)
        with st.spinner("Thinking..."):
            response = claude.messages.create(
                model="claude-opus-5-5",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            )
        answer = next(b.text for b in response.content if b.type == "text")

        # Step 4: show the answer and where it came from
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for c, s in relevant:
            with st.expander(s["source"]):
                st.write(c)
