from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# RAG prompt templates for different scenarios

# Standard RAG prompt for general question answering
RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that answers questions based on the provided documents. 
    Use only information from the documents to answer the question. 
    If you don't know the answer or the information is not in the documents, say so clearly.
    Always cite the sources by mentioning which Document and page your information comes from.
    Be concise, clear, and helpful."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("system", "Relevant documents for this question:\n\n{context}")
])

# Prompt for summarizing a set of documents
DOCUMENT_SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that summarizes information from documents.
    Create a concise summary of the key points from the provided documents.
    Focus on the most important facts and insights.
    Organize the information logically and maintain factual accuracy."""),
    ("system", "Documents to summarize:\n\n{context}"),
    ("human", "Please provide a summary of these documents."),
])

# Prompt for answering questions with step-by-step reasoning
REASONING_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that answers questions based on the provided documents.
    Use only information from the documents to answer the question.
    If you don't know the answer or the information is not in the documents, say so clearly.
    Use step-by-step reasoning to reach your answer.
    Always cite the sources by mentioning which Document and page your information comes from."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
    ("system", "Relevant documents for this question:\n\n{context}"),
    ("system", "First, analyze the question. Then extract relevant information from the documents. Finally, synthesize a clear answer.")
])

# Prompt for extracting specific information from documents
INFORMATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that extracts specific information from documents.
    Focus only on extracting the exact information requested.
    If the information is not present in the documents, say so clearly.
    Provide the exact text from the document where possible and cite the source."""),
    ("system", "Documents to search:\n\n{context}"),
    ("human", "Please extract the following information: {extraction_query}"),
])
