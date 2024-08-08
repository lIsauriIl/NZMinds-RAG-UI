import re
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings


# List of vectorstore directory names
dir_list = ['overview', 'services', 'industries', 'technologies', 'portfolio', 
            'web', 'mobile', 'ui-ux', 'saas', 'blockchain', 'ai-ml']


# List of URLs to be processed
urls = [
    'https://nzminds.com/',
    ['https://nzminds.com/software-development-services',
    'https://nzminds.com/ai-ml-application-development-services',
    'https://nzminds.com/cloud-migration-services',
    'https://nzminds.com/web-application-development-services',
    'https://nzminds.com/mobile-application-development-services',
    'https://nzminds.com/custom-software-development-services',
    'https://nzminds.com/saas-application-development-services',
    'https://nzminds.com/software-qa-testing-services',
    'https://nzminds.com/blockchain-development-services',
    'https://nzminds.com/cyber-security-applications-development-services',
    'https://nzminds.com/ui-ux-design-and-development-services',
    'https://nzminds.com/machine-learning-development-services',
    'https://nzminds.com/devops-consulting-services',
    'https://nzminds.com/database-management-services',
    'https://nzminds.com/software-maintenance-services',
    'https://nzminds.com/it-support-services'],
    ['https://nzminds.com/industries',
    'https://nzminds.com/healthcare-software-development',
    'https://nzminds.com/fintech-software-development',
    'https://nzminds.com/ecommerce-software-development',
    'https://nzminds.com/edtech-software-development',
    'https://nzminds.com/realestate-software-development',
    'https://nzminds.com/cybersecurity-software-development'],
    'https://nzminds.com/technologies',
    'https://nzminds.com/portfolio/',
    ['https://nzminds.com/portfolio/web-1',
    'https://nzminds.com/portfolio/web-2',
    'https://nzminds.com/portfolio/web-3',
    'https://nzminds.com/portfolio/web-4'],
    ['https://nzminds.com/portfolio/app-1',
    'https://nzminds.com/portfolio/app-2',
    'https://nzminds.com/portfolio/app-3',
    'https://nzminds.com/portfolio/app-4'],
    ['https://nzminds.com/portfolio/ui-1',
    'https://nzminds.com/portfolio/ui-2',
    'https://nzminds.com/portfolio/ui-3',
    'https://nzminds.com/portfolio/ui-4'],
    ['https://nzminds.com/portfolio/saas-1',
    'https://nzminds.com/portfolio/saas-2',
    'https://nzminds.com/portfolio/saas-3',
    'https://nzminds.com/portfolio/saas-4'],
    ['https://nzminds.com/portfolio/blockchain-1',
    'https://nzminds.com/portfolio/blockchain-2',
    'https://nzminds.com/portfolio/blockchain-3',
    'https://nzminds.com/portfolio/blockchain-4'],
    ['https://nzminds.com/portfolio/ai-ml-1',
    'https://nzminds.com/portfolio/ai-ml-2',
    'https://nzminds.com/portfolio/ai-ml-3',
    'https://nzminds.com/portfolio/ai-ml-4']
]


# Function to merge hyphenated words split by new lines
def merge_hyphenated_words(text):
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# Function to replace single new lines with spaces
def fix_newlines(text):
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


# Function to reduce multiple new lines to a single new line
def remove_multiple_newlines(text):
    return re.sub(r"\n{2,}", "\n", text)


# Function to replace multiple spaces with a new line
def remove_duplicate_spaces(text):
    return re.sub(r'\s{2,}', '\n', text)


# Function to clean the text by applying a series of cleaning functions
def clean_text(text):
    cleaning_functions = [merge_hyphenated_words, remove_multiple_newlines, remove_duplicate_spaces]
    for cleaning_function in cleaning_functions:
        text = cleaning_function(text)
    return text


# Function to split text into document chunks
def text_to_docs(text, metadata):
    doc_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200, add_start_index=True)
    text_splitter = SemanticChunker(embeddings=OllamaEmbeddings(model='mxbai-embed-large'), breakpoint_threshold_type='percentile')
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        doc = Document(page_content=chunk, metadata=metadata)
        doc_chunks.append(doc)
    return doc


# Function to clean a list of Document objects
def clean_docs(list_of_docs:list[Document]) -> list[Document]:
    cleaned_doc_list = []
    for doc in list_of_docs:
        content = doc.page_content
        metadata = doc.metadata
        modified_content = clean_text(content)[:-1252]
        index = modified_content.find("TOUCH")
        cleaned_content = modified_content[index+7:]
        cleaned_content = cleaned_content[280:]
        cleaned_doc = Document(page_content=cleaned_content, metadata=metadata)
        cleaned_doc_list.append(cleaned_doc)
    return cleaned_doc_list


# Main pipeline function to process and embed documents
def embedding_pipeline(listofurls, dir_list):
    for element, dir in zip(listofurls, dir_list):
        if type(element) is str:
            docs = WebBaseLoader(web_path=element).load()
        elif type(element) == list:
            docs = WebBaseLoader(web_paths=element).load()
        cleaned_docs = clean_docs(list_of_docs=docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=768 if type(element) is str
                                                else (1024 if len(element) < 5 else 2048),
                                                 chunk_overlap=77 if type(element) is str
                                                  else (100 if len(element) < 5 else 200),
                                                   add_start_index=True)
        semantic_splitter = SemanticChunker(embeddings=OllamaEmbeddings(model='mxbai-embed-large'), breakpoint_threshold_type='percentile')
        chunks = splitter.split_documents(cleaned_docs)
        count=0
        for chunk in chunks:
            print(chunk.page_content)
            print('\n')
            print(count, "CHUNK PRINTED---------------------------------------------------------------------------------------------")
            count+=1
        db = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings(model='mxbai-embed-large'),
                                   persist_directory=dir)
        print("EMBEDDING STORED")


# Run the embedding pipeline
embedding_pipeline(listofurls=urls, dir_list=dir_list)
