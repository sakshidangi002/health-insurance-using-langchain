from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import streamlit as st
from PyPDF2 import PdfReader



load_dotenv()
llm=ChatOllama(model="gemma:2b",
goal="your are a health insurance policy expert")


#file uploader
def document_comparison_section():
    st.header("Compare Two Health Insurance Policy Documents")
    uploaded_files = st.file_uploader(
        "Upload two insurance policy documents (PDF or TXT)", 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and 2<= len(uploaded_files) <= 5:
        docs = []
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            else:
                # For TXT files, decode directly
                text = file.read().decode("utf-8")
            docs.append(text)

        # text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunked_docs=[text_splitter.split_text(doc) for doc in docs]

        # embeddings
        embeddings = OllamaEmbeddings(model="all-minilm")
        vectors = [FAISS.from_texts(chunks, embeddings) for chunks in chunked_docs]

        # LLM comparison
        model = ChatOllama(model="gemma:2b")
        compare_prompt = "Compare the following health insurance policy documents:\n\n"
        for i, doc in enumerate(docs, start=1):
            compare_prompt += f"Document {i}:\n{doc[:2000]}\n\n" 

        compare_prompt += (
            "Please summarize the main differences between these policies in terms of:\n"
            "- Coverage\n"
            "- Deductibles\n"
            "- Co-payments\n"
            "- Exclusions\n"
            "- Special benefits\n\n"
            "Also, highlight which policy might be best suited for different types of customers."
        )
        response = model.invoke(compare_prompt)

        st.subheader("Comparison Output")
        st.markdown(response.content if hasattr(response, "content") else str(response))

    elif uploaded_files:
        st.info("Please upload exactly two policy documents to compare.")


def main():
    document_comparison_section()

if __name__ == "__main__":
    main()


   
