# Corporate PDF Analyzer

A professional web application that analyzes PDF documents using OpenAI's GPT-4 to provide detailed summaries and insights in JSON format.

## Features

- 🚀 Modern web interface with Bootstrap styling
- 📊 Real-time progress tracking
- 📝 Live analysis logging
- 🔄 Drag and drop file upload
- 📱 Responsive design
- 🔍 Detailed PDF analysis using GPT-4
- 💾 JSON output format
- 🔒 Secure file handling

## Prerequisites

- Python 3.8+
- OpenAI API key
- Flask
- Modern web browser

## Installation

1. Clone the repository:

:
bash
git clone [your-repository-url]
cd pdf-synth
```


2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```
3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Project Structure
```
pdf-synth/
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
├── .env # Environment variables
├── .gitignore # Git ignore rules
├── templates/ # HTML templates
│ └── index.html # Main interface
├── static/ # Static assets
│ └── style.css # CSS styles
├── uploads/ # Temporary PDF storage
├── outputs/ # Analysis output files
└── logs/ # Application logs
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```bash
http://localhost:5000
```

3. Upload a PDF file:
   - Drag and drop a PDF file onto the upload area
   - Or click "Browse Files" to select a PDF

4. Monitor the analysis:
   - Watch real-time progress bar
   - View detailed logs in the console
   - Download results in JSON format

## Key Components

### PDF Processing
- Secure file upload handling
- PDF text extraction using PyPDF2
- Automatic cleanup of temporary files

### AI Analysis
- GPT-4 powered content analysis
- Structured JSON output
- Detailed processing logs

### User Interface
- Progress tracking with visual feedback
- Real-time log console with timestamps
- Professional corporate styling
- Mobile-responsive design

## Security Features

- Secure filename handling
- File type verification using python-magic
- Maximum file size limit (16MB)
- Automatic file cleanup
- Comprehensive error handling and logging

## Logging System

The application includes detailed logging:
- File operations
- Analysis steps
- Error tracking
- User actions

Logs are stored in the `logs` directory with automatic rotation:
- Maximum log size: 10MB
- Backup count: 5 files
- Debug level logging to file
- Info level logging to console

## Environment Variables

Required environment variable:
- `OPENAI_API_KEY`: Your OpenAI API key

## Error Handling

The application handles various errors:
- Invalid file types
- File size limits
- API failures
- Processing errors
- Missing API keys

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Future Enhancements

- User authentication system
- Analysis history tracking
- Advanced analytics dashboard
- Batch processing capability
- Additional export formats
- Team collaboration features


---

**Note**: This application requires an active OpenAI API key 