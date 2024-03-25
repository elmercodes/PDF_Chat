import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
import tempfile


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0.4, model="gpt-3.5-turbo")


#sidebar contents
with st.sidebar:
    st.title("PDF Chat Expert")
    st.markdown('''
    ## About
    This chat interface will allow you to converse with numerouse PDFs of your choosing!
    Simply upload your PDFs(up to 5) then ask questions, simple as that
    
                ''')
    add_vertical_space(5)
    st.write("Made by Elmer Flores, loser")
    
    
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_paths.append(os.path.join(temp_dir, uploaded_file.name))
    return saved_paths
    
    
def main():
    st.header("Chat With PDFs")
    pdfs = st.file_uploader("Upload up to 5 PDF files", type="pdf", accept_multiple_files=True)
    
    if pdfs:
        saved_paths = save_uploaded_files(pdfs)
        reader = SimpleDirectoryReader(input_files=saved_paths)
        data = reader.load_data()
        index = VectorStoreIndex.from_documents(data)
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=3)
        
        
        query = st.text_input("Ask questions regarding/relating to your PDFs")
        if query:
            response = query_engine.query(query)
            answer = str(response)
            st.write(answer)

            for node in response.source_nodes:
                st.write("-------")

                text_fmt = node.node.get_content().strip().replace("\n", " ")[:100]
                st.write(f"Reference:\t {text_fmt} ...")

                metadata = node.node.metadata.items()
                if len(metadata) >= 2:
                    st.write(f"Location:\t {dict(list(metadata)[:2])}")
                else:
                    st.write("Location:\t No metadata available")

                st.write(f"Confidence Score:\t {node.score:.3f}")

    
    
# if __name__ == "__main__":
#     main()
    
    
