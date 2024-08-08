import streamlit as st
from nzminds_rag import * 

# Function to trim the chat history. By default, it is untrimmed each time program is rerun so we have to trim each time
def trim_history(history):
    trimmed_history = []
    while len(trimmed_history) < 5:    
        for message in history:
            trimmed_history.append(message)
    return trimmed_history


# Page Configurations and main features
st.set_page_config(
    page_title='Chat with NZMinds chatbot',
    page_icon='🧠',
    menu_items={
        "About": 'https://nzminds.com'
    }
)

st.title('Chat with NZMinds chatbot')
st.subheader('Hi there!')

# The first message that the AI gives, before the convo starts
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'ai', 'content': 'Hi there! I am a chatbot representing NextZen Minds. How can I help you?'}]

model = create_model('phi3')

trimmed_history = trim_history(st.session_state.messages)

sys_prompt_text = (
        "You are an AI chatbot that represents NextZen Minds. You help "
        "users in question answering tasks. Use the given context below to "
        "answer the user's question. When answering, "
        "keep your answers short. Give answers in a pleasing and informational "
        "tone. When asked to list or mention things, ONLY give those items, and "
        "don't give explanations unless asked otherwise. If the answer isn't given "
        "in the context, say you don't know. Don't give explanations "
        "that aren't inside the context. Remember, ONLY answer from the context. "
        "Give a natural, human-like response. Don't give responses containing things like "
        "'according to the given context', 'as mentioned', 'as stated', etc. Just "
        "give the answer directly. "
        "Additionally, you will be provided extra context as supplementary information. "
        "If there is a clash between the retrieved context and the provided extra context, "
        "prioritise the extra context. Furthermore, you will be provided chat history. "
        "In the event that the user asks a question that references previous parts "
        "of the conversation, go through the chat history, and taking into account the "
        "chat history, context, and extra context, answer the question. "
        "Don't say things like 'Extra context (not part of original answer)', or "
        "'According to the chat history' either."
        "\n\n"
        "Context: {context}\n"
        "Extra context: {extra_context}\n"
        "Chat history: {chat_history}\n"
        "Keep in mind this as well, when they ask for the location of NZMinds: "
        """
            Singapore Head Office:
            105 Cecil Street
            #22-00 The Octagon (Suite 2210)
            Singapore 069534
            Ph: +65 81574799 / +1 8884210410

            India Office:
            Ambuja Ecostation PLOT NO. 7, BLOCK-BP
            Unit : 1501 & 1504
            SECTOR V, SALT LAKE, KOLKATA-700091
            Ph: +1 8884210410

            Netherlands Office:
            117, Sciencepark, Mendelweg 32,
            Leiden, 2333 CS
            Ph: +91 9148140224 / +1 8884210410\n
            """
            "For email, NZMinds can be contacted via business@nzminds.com or info@nzminds.com."
    )


# Loop to display all buttons on the sidebar, making them clickable
with st.sidebar:
    st.subheader('Categories')
    st.write('Note: starts with Overview by default')
    for cat in category_dict.keys():
        # If button is pressed, load the corresponding retriever into session state
        if st.button(cat):
            with st.spinner('Loading vectorstore...'):
                st.session_state['retriever'] = create_retriever(category_name=cat)
            st.write(f"Current topic: {cat}")
            st.session_state['extra_ctx'] = extra_context_dict[cat]

# Create RAG chain based on retriever, and based on Overview upon first initialisation
if 'retriever' in st.session_state:
    rag_chain = create_rag_chain(model=model, prompt_template=create_prompt_template(sys_prompt_text), retriever=st.session_state['retriever'])
else:
    rag_chain = create_rag_chain(model=model, prompt_template=create_prompt_template(sys_prompt_text), retriever=create_retriever(category_name='Overview'))

# Load the Overview extra context upon first initialisation
if 'extra_ctx' not in st.session_state:
    st.session_state['extra_ctx'] = extra_context_dict['Overview']

# Printing the entire message history, before asking the next question to the AI
for message in st.session_state.messages:
    with st.chat_message(name=message['role']):
        st.write(message['content'])

# User asks a question
with st.chat_message(name='human'):
    query = st.chat_input('Enter your query')
    # By default, the chat input is set as None, so in order to do something to the input, user needs to write text
    if query is not None:
        st.markdown(query)

# AI responds
with st.chat_message(name='ai'):
    if query is not None:
        # Loading while carrying out the process (can take more than 7 min)
        with st.spinner('Generating response - this may take a while...'):    
            response = rag_chain.invoke({'input': query, 'extra_context': st.session_state['extra_ctx'],
                                     'chat_history': trimmed_history})['answer']
            
            st.markdown(response)
        # Adding chat history. This specific format is required so that while printing history, we can access each key
        # to use in their respective containers (chat_message('ai') for example)
        st.session_state.messages.extend([{'role': 'human', 'content': query},
                                          {'role': 'ai', 'content': response}])

