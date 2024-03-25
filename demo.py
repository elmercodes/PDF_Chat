import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
import tempfile


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0.4, model="gpt-3.5-turbo")
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)


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
    
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your documents!"}
        ]
    
    if pdfs:
        saved_paths = save_uploaded_files(pdfs)
        reader = SimpleDirectoryReader(input_files=saved_paths)
        data = reader.load_data()
        index = VectorStoreIndex.from_documents(data)
        
        chat_engine = index.as_chat_engine(
            streaming=True, 
            similarity_top_k=3,
            chat_mode="react",
            memory=memory,
            llm=llm,
            context_prompt=(
                "You are a chatbot, able to have interactions regarding the documents given as well as related questions."
                "\nInstruction: Use the previous chat history, or given documents, to interact and help the user."
            ),
            verbose=False,
            )

        if prompt := st.chat_input("Ask questions regarding/relating to your PDFs"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message) # Add response to message history
        
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        # query = st.text_input("Ask questions regarding/relating to your PDFs")
        # if query:
        #     response = query_engine.query(query)
        #     answer = str(response)
        #     st.write(answer)

        #     for node in response.source_nodes:
        #         st.write("-------")

        #         text_fmt = node.node.get_content().strip().replace("\n", " ")[:100]
        #         st.write(f"Reference:\t {text_fmt} ...")

        #         metadata = node.node.metadata.items()
        #         if len(metadata) >= 2:
        #             st.write(f"Location:\t {dict(list(metadata)[:2])}")
        #         else:
        #             st.write("Location:\t No metadata available")

        #         st.write(f"Confidence Score:\t {node.score:.3f}")

    
    

    
