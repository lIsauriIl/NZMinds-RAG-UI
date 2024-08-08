# NZMinds-RAG-UI

# Introduction
A RAG chatbot implementation via LangChain and Streamlit that answers user queries based on the software company NextZen Minds. The under-the-hood logic is taken care of by LangChain and Ollama, and the LLM used is Phi3, which has 3 billion parameters. The UI is constructed via Streamlit.

# Installation and setup
Git clone the repo via ```git clone https://github.com/lIsauriIl/NZMinds-RAG-UI```, or you can manually install the files yourself. Make sure to do that in the same repository. Optionally, you can initialise a venv, but it's highly recommended.

Carry out the following command: ```pip install -r requirements.txt```.

# Running the program
First, run nzminds_chunk_vecdb.py. This will take a while since it's done on local hardware.

Then on the command line, run ```streamlit run chatbot_app.py```
