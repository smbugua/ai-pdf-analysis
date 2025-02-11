import os
from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
from werkzeug.utils import secure_filename
import PyPDF2
import openai
from dotenv import load_dotenv
import json
import magic
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import tiktoken
import pickle

# Load environment variables
load_dotenv()

# Configure logging
def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('pdf_analyzer')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler (rotating file handler to manage log size)
    file_handler = RotatingFileHandler(
        'logs/pdf_analyzer.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['VECTOR_STORE'] = 'vector_store'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_STORE'], exist_ok=True)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found")

# Initialize chat model and memory
chat_model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
)

# Initialize conversation memories dictionary
conversation_memories = {}

def get_or_create_memory(session_id):
    if session_id not in conversation_memories:
        conversation_memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
        )
    return conversation_memories[session_id]

def get_vector_store_path(filename):
    """Get the path for vector store files"""
    base_name = os.path.splitext(filename)[0]
    vector_store_path = os.path.join(app.config['VECTOR_STORE'], f"{base_name}_store")
    logger.debug(f"Vector store path: {vector_store_path}")
    return vector_store_path

def save_vector_store(vectorstore, filepath):
    """Save the FAISS vector store"""
    vectorstore.save_local(filepath)

def load_vector_store(filepath):
    """Load the FAISS vector store"""
    embeddings = OpenAIEmbeddings()
    if os.path.exists(filepath):
        return FAISS.load_local(filepath, embeddings, allow_dangerous_deserialization=True)
    return None

def create_chat_prompt():
    """Create the chat prompt template"""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful assistant that answers questions based on the provided document. "
            "Use the context provided to answer questions accurately and concisely. "
            "If you don't know the answer or it's not in the context, say so."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=(
            "Context: {context}\n"
            "Question: {question}"
        )),
    ])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def extract_text_from_pdf(pdf_path):
    logger.info(f"Starting text extraction from PDF: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            logger.debug(f"PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                logger.debug(f"Processing page {page_num}/{total_pages}")
                text += page.extract_text()
        
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        raise

def count_tokens(text):
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def create_text_chunks(text):
    """Split text into chunks using LangChain's text splitter."""
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=count_tokens,
    )
    chunks = text_splitter.split_text(text)
    logger.debug(f"Created {len(chunks)} chunks")
    return chunks

def process_with_rag(text, filename):
    """Process large documents using RAG approach and save to vector store."""
    logger.info("Starting RAG processing")
    try:
        # Create chunks
        chunks = create_text_chunks(text)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(
            chunks,
            embeddings,
            metadatas=[{"source": filename} for _ in chunks]
        )
        
        # Save vector store
        vector_store_path = get_vector_store_path(filename)
        save_vector_store(vectorstore, vector_store_path)
        
        # Process chunks and aggregate results
        all_analyses = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_analysis = analyze_chunk_with_openai(chunk)
            all_analyses.append(json.loads(chunk_analysis))
        
        # Combine analyses
        combined_analysis = combine_analyses(all_analyses)
        logger.info("Successfully completed RAG processing")
        return json.dumps(combined_analysis)
    except Exception as e:
        logger.error(f"Error in RAG processing: {str(e)}", exc_info=True)
        raise

def analyze_chunk_with_openai(text):
    """Analyze a single chunk with OpenAI."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes document chunks and provides detailed summaries in a structured format."},
                {"role": "user", "content": f"Please analyze this text chunk and provide a detailed summary including main topics, key points, and important details. Format the response as JSON with keys: 'main_topics', 'key_points', 'details': {text}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in chunk analysis: {str(e)}", exc_info=True)
        raise

def combine_analyses(analyses):
    """Combine multiple chunk analyses into a single coherent analysis."""
    logger.info("Combining chunk analyses")
    try:
        # Prepare combined data
        all_topics = []
        all_key_points = []
        all_details = []
        
        for analysis in analyses:
            all_topics.extend(analysis.get('main_topics', []))
            all_key_points.extend(analysis.get('key_points', []))
            all_details.extend(analysis.get('details', []))
        
        # Use OpenAI to synthesize the combined information
        synthesis_prompt = {
            "topics": all_topics,
            "key_points": all_key_points,
            "details": all_details
        }
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that synthesizes multiple document analyses into a coherent summary."},
                {"role": "user", "content": f"Please synthesize these analyses into a single coherent summary. Remove duplicates and organize the information logically. Format the response as JSON with keys 'main_topics', 'key_points', 'details', and 'executive_summary': {json.dumps(synthesis_prompt)}"}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error combining analyses: {str(e)}", exc_info=True)
        raise

# Modify the analyze_text_with_openai function to handle large documents
def analyze_text_with_openai(text, filename):
    logger.info("Starting text analysis")
    logger.debug(f"Text length: {len(text)} characters")
    
    try:
        # Check if text is too large (e.g., more than 4000 tokens)
        if count_tokens(text) > 4000:
            logger.info("Large document detected, using RAG processing")
            return process_with_rag(text, filename)
        else:
            logger.info("Document size within limits, using standard processing")
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides detailed summaries in a structured format."},
                    {"role": "user", "content": f"Please analyze this text and provide a detailed summary including main topics, key points, and important details. Format the response as JSON with appropriate keys and values: {text}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}", exc_info=True)
        raise

@app.route('/', methods=['GET'])
def index():
    logger.info("Accessed home page")
    return render_template('index.html')

@app.route('/chat', methods=['GET'])
def chat_page():
    filename = request.args.get('filename')
    if not filename:
        return redirect('/')
    return render_template('chat.html')

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    logger.info("PDF analysis request received")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        logger.info(f"Processing file: {filename}")
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        logger.debug(f"File saved to: {pdf_path}")

        # Verify file is actually a PDF
        file_type = magic.from_file(pdf_path, mime=True)
        if file_type != 'application/pdf':
            logger.warning(f"Invalid PDF file type detected: {file_type}")
            os.remove(pdf_path)
            return jsonify({'error': 'Invalid PDF file'}), 400

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Analyze text using OpenAI
        analysis_json = analyze_text_with_openai(text, filename)
        
        # Save JSON output
        json_filename = f"{filename[:-4]}_analysis.json"
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], json_filename)
        with open(json_path, 'w') as f:
            f.write(analysis_json)
        logger.debug(f"JSON analysis saved to: {json_path}")
        
        # # Save text output
        # text_filename = f"{filename[:-4]}_analysis.txt"
        # text_path = os.path.join(app.config['OUTPUT_FOLDER'], text_filename)
        # with open(text_path, 'w') as f:
        #     f.write(json.loads(analysis_json)['summary'])
        # logger.debug(f"Text analysis saved to: {text_path}")

        # Clean up uploaded PDF
        os.remove(pdf_path)
        logger.debug(f"Cleaned up uploaded file: {pdf_path}")

        logger.info(f"Successfully analyzed PDF: {filename}")
        return jsonify({
            'success': True,
            'json_file': json_filename,
            #'text_file': text_filename,
            'analysis': json.loads(analysis_json)
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        # Clean up uploaded file if it exists
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.debug(f"Cleaned up uploaded file after error: {pdf_path}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        session_id = data.get('session_id')
        filename = data.get('filename')

        if not all([question, session_id, filename]):
            logger.warning('Missing required parameters')
            return jsonify({'error': 'Missing required parameters'}), 400

        logger.info(f'Received chat request for session: {session_id}, filename: {filename}')

        # Load vector store
        vector_store_path = get_vector_store_path(filename)
        logger.debug(f'Loading vector store from path: {vector_store_path}')
        vectorstore = load_vector_store(vector_store_path)
        
        if not vectorstore:
            logger.warning('No document context found for the given filename')
            return jsonify({'error': 'No document context found'}), 404

        # Get relevant documents
        logger.info('Performing similarity search in vector store')
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join(doc.page_content for doc in docs)

        if not context:
            logger.warning('No relevant context found for the question')
            return jsonify({'error': 'No relevant context found'}), 404

        # Get or create memory for this session
        memory = get_or_create_memory(session_id)

        # Create prompt
        prompt = create_chat_prompt()

        # Get chat history
        chat_history = memory.load_memory_variables({})

        # Log the context and question before formatting
        logger.debug(f'Context: {context}')
        logger.debug(f'Question: {question}')

        # Use the hardcoded context
        context = """
- The document discusses the terms and conditions of a service agreement between two parties.
- It outlines the responsibilities of each party, the duration of the contract, and the payment terms.
- Additionally, it includes clauses on confidentiality, termination, and dispute resolution.
"""

        # Log the hardcoded context
        logger.debug(f'Hardcoded Context: {context}')

        # Format the prompt with actual context and question
        formatted_messages = prompt.format_messages(
            context=context,
            question=question,
            chat_history=chat_history.get("chat_history", [])
        )

        # Generate response
        logger.info('Generating response using chat model')
        response = chat_model.invoke(formatted_messages)

        if not response.content:
            logger.warning('No response generated by chat model')
            return jsonify({'error': 'No response generated'}), 500

        # Update memory
        memory.save_context(
            {"input": question},
            {"output": response.content}
        )

        logger.info('Successfully generated response')
        return jsonify({
            'response': response.content,
            'context': context
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.errorhandler(413)
def request_entity_too_large(error):
    logger.warning("File upload exceeded size limit")
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting PDF Analyzer application")
    app.run(debug=True) 

# Hardcoded context for testing
context = """
- The document discusses the terms and conditions of a service agreement between two parties.
- It outlines the responsibilities of each party, the duration of the contract, and the payment terms.
- Additionally, it includes clauses on confidentiality, termination, and dispute resolution.
"""

# Log the hardcoded context
logger.debug(f'Hardcoded Context: {context}') 