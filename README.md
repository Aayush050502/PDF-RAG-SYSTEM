

This project implements a Retrieval-Augmented Generation (RAG) pipeline to process multilingual PDFs. It enables users to extract, query, and summarize information from PDFs in various languages, including Hindi, English, Bengali, and Chinese. The application leverages Streamlit for interaction and Google Generative AI for embeddings and conversational responses.

Features
Multilingual Support: Process PDFs in multiple languages (scanned and digital).
Text Extraction:
OCR for scanned documents.
Standard extraction for digital PDFs.
RAG Pipeline:
Query-based document retrieval.
Context-aware answers using vector-based search.
Scalable for large datasets (up to 1TB).
Chat Functionality:
Conversational interface with query decomposition and memory.
Hybrid search combining keyword and semantic techniques.
Model Efficiency:
Optimized for small LLMs and embeddings.
Installation
Clone the repository:

bash
Copy code
git clone <repository_url>
cd <repository_folder>
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up Google API credentials:

Add your GOOGLE_API_KEY to an .env file.
Run the application:

bash
Copy code
streamlit run app.py
How to Use
Upload PDFs:
Drag and drop PDFs (scanned or digital) into the sidebar.
Process Files:
Click the "Submit & Process" button to extract and vectorize text.
Query PDF Content:
Enter a question in the main input field.
View detailed responses derived from the uploaded PDFs.
Key Components
Text Processing:
Text is extracted and chunked for efficient querying.
Vector Store:
Built using FAISS with embeddings from Google Generative AI.
Conversational Engine:
Uses LangChain to generate responses based on extracted context.
Evaluation Metrics
Query Relevance: Accurate results matching user intent.
Latency: Low response time.
Fluency: Clear, concise, and well-organized answers.
Scalability: Handles large datasets efficiently.
Deliverables
Functional RAG system with query and summarization features.
Documentation for architecture, user guide, and performance metrics.
