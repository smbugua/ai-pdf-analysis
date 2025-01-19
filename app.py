import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import openai
from dotenv import load_dotenv
import json
import magic
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

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

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found")

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

def analyze_text_with_openai(text):
    logger.info("Starting OpenAI analysis")
    logger.debug(f"Analyzing text of length: {len(text)} characters")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides detailed summaries in a structured format."},
                {"role": "user", "content": f"Please analyze this text and provide a detailed summary including main topics, key points, and important details. Format the response as JSON with appropriate keys and values: {text}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        logger.info("Successfully received OpenAI analysis")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI analysis: {str(e)}", exc_info=True)
        raise

@app.route('/', methods=['GET'])
def index():
    logger.info("Accessed home page")
    return render_template('index.html')

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
        analysis_json = analyze_text_with_openai(text)
        
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