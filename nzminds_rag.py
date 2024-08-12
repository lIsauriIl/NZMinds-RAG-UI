from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.messages import HumanMessage, AIMessage
import os, getpass
import extra_context

# Dictionary containing category names and corresponding directory paths
category_dict = {
    'Overview': 'overview',
    'Services': 'services',
    'Industries': 'industries',
    'Technologies': 'technologies',
    'Portfolio': 'portfolio',
    'Web Development Portfolio': 'web',
    'Mobile Development Portfolio': 'mobile',
    'UI/UX Development Portfolio': 'ui-ux',
    'SaaS Development Portfolio': 'saas',
    'Blockchain Development Portfolio': 'blockchain',
    'AI/ML Development Portfolio': 'ai-ml'
}

# List of portfolio categories
portfolio_list = ['Web Development Portfolio', 'Mobile Development Portfolio',
                  'UI/UX Development Portfolio', 'SaaS Development Portfolio',
                  'Blockchain Development Portfolio', 'AI/ML Development Portfolio']

# Dictionary containing extra context information for each category
extra_context_dict = {
    'Overview': extra_context.overview,
    'Services': extra_context.services,
    'Industries': extra_context.industries,
    'Technologies': extra_context.technologies,
    'Portfolio': extra_context.portfolio,
    'Web Development Portfolio': extra_context.web,
    'Mobile Development Portfolio': extra_context.mobile,
    'UI/UX Development Portfolio': extra_context.ui_ux,
    'SaaS Development Portfolio': extra_context.saas,
    'Blockchain Development Portfolio': extra_context.blockchain,
    'AI/ML Development Portfolio': extra_context.ai_ml,
    'n': extra_context.portfolio
}


# Function to create a retriever for a given category
def create_retriever(category_name: str) -> VectorStoreRetriever:
    dir_path = category_dict[category_name]
    embed = OllamaEmbeddings(model='mxbai-embed-large')
    db = Chroma(persist_directory=dir_path, embedding_function=embed)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 7})
    model = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base')
    compressor = CrossEncoderReranker(model=model, top_n=4 if category_name in portfolio_list else 3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    return compression_retriever


# Function to create a chat model
def create_model(model_name: str = 'llama2') -> ChatOllama:
    model = ChatOllama(model=model_name)
    return model


# Function to create a prompt template
def create_prompt_template(prompt_text: str) -> ChatPromptTemplate:
    template = ChatPromptTemplate.from_messages([
        ('system', prompt_text),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])
    return template


# Function to create a RAG (Retrieval-Augmented Generation) chain
def create_rag_chain(model: ChatOllama, prompt_template: ChatPromptTemplate, retriever: VectorStoreRetriever):
    qa_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)
    return rag_chain


# Function to run the chatbot
def run_chatbot():
    """
    Run the chatbot to interact with users, providing answers based on retrieved context,
    extra context, and chat history.

    The function operates in an infinite loop, allowing the user to choose categories
    and ask questions. The chatbot uses a Retrieval-Augmented Generation (RAG) chain
    to fetch and combine documents for generating responses. The chat history is 
    maintained to provide context for ongoing conversations.

    The chatbot can be exited by entering "exit" or navigating back in the menu.
    """
    chat_history = []  # List to maintain chat history
    model = create_model('phi3')  # Initialize the chat model with the specified model name

    # System prompt text that defines the behavior and guidelines for the chatbot
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

    # Create a prompt template using the system prompt text
    prompt_template = create_prompt_template(prompt_text=sys_prompt_text)

    while True:
        # Initial greeting and category display
        print('Hi there! We are NextZen Minds. What would you like to talk about?\n')
        for key in category_dict.keys():
            if key == 'Web Development Portfolio':
                break
            print(key)
        
        # Get the user's choice of category
        chosen_main_category = str(input('Choose category, or "exit" to exit: '))
        if chosen_main_category == 'exit':
            break

        if chosen_main_category == 'Portfolio':
            # If the user chooses "Portfolio", ask for a specific sub-category
            print('Do you want to talk about a particular category in our portfolio? Here are the categories:\n')
            for cat in portfolio_list:
                print(cat)
            print('\n')
            portfolio_input = str(input('Pick one of the categories above, or "n" to talk in general, or "back" to go back: '))
            if portfolio_input == 'back':
                break
            
            # Determine the retriever based on user input
            if portfolio_input != 'n':
                retriever = create_retriever(category_name=portfolio_input)
            else:
                retriever = create_retriever(category_name=chosen_main_category)

            # Create the RAG chain for handling queries
            rag_chain = create_rag_chain(model=model, prompt_template=prompt_template, retriever=retriever)
            
            # Handle user queries within the selected category
            while True:
                user_query = str(input('Enter query, or "back" to go back: '))
                if user_query == 'back':
                    break
                
                # Invoke the RAG chain to get the response
                response = rag_chain.invoke({
                    'input': user_query,
                    'extra_context': extra_context_dict[portfolio_input],
                    'chat_history': chat_history
                })['answer']
                
                print(response)
                # Update chat history with the user's query and AI's response
                chat_history.extend([HumanMessage(content=user_query), AIMessage(content=response)])
                
                # Keep the chat history manageable by limiting it to the last 5 exchanges
                if len(chat_history) > 5:
                    del chat_history[0]
                    del chat_history[0]
        else:
            # Handle other main categories similarly
            retriever = create_retriever(category_name=chosen_main_category)
            rag_chain = create_rag_chain(model=model, prompt_template=prompt_template, retriever=retriever)
            while True:
                user_query = str(input('Enter query, or "back" to go back: '))
                if user_query == 'back':
                    break
                
                response = rag_chain.invoke({
                    'input': user_query,
                    'extra_context': extra_context_dict[chosen_main_category],
                    'chat_history': chat_history
                })['answer']
                
                print(response)
                chat_history.extend([HumanMessage(content=user_query), AIMessage(content=response)])
                
                if len(chat_history) > 5:
                    del chat_history[0]
                    del chat_history[0]

if __name__ == '__main__':
    run_chatbot()

