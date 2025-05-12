# MangaWhisperer

MangaWhisperer is a Python application that lets you upload a Japanese manga page (image) and have the dialogue read back to you as speech. It combines OCR, speech-recognition, and text-to-speech models behind a simple Flask-powered web API.

The project can be used with either *local models* or calling *APIs*, edit `server.py` to change the behavior of the program.


## Features

- **Image Upload & Processing**  
  Accepts manga page images, segments text regions, and extracts Japanese dialogue via an OCR pipeline.  
- **Speech Recognition**  
  Uses OpenAI’s Whisper model to refine and transcribe extracted text.  
- **Text-to-Speech**  
  Converts the final Japanese transcript into spoken audio.  
- **Web API & UI**  
  Exposes endpoints via Flask for programmatic use, plus a minimal web frontend to interactively upload pages.

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/ThiagoLira/MangaWhisperer.git
   cd MangaWhisperer
   ```
2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**  
    If you opt to use OpenAI APIs please create a `.env` file with OPENAI_API_KEY defined.


## Usage

- **Run the server**  
  ```bash
  bash run_server.sh
  ```
  This will start the Flask app (by default at `http://127.0.0.1:5000/`).  
- **API endpoint**  
  - `POST /process` — multipart form upload of a manga image; returns JSON with the transcript and a link to the generated audio.  
- **Web UI**  
  Open `http://127.0.0.1:5000/` in your browser to upload pages and listen to the results.

## Project Structure

```
.
├── image_processor.py   # handles image loading, OCR, and pre-processing
├── utils.py             # helper functions for text cleaning and file management
├── server.py            # Flask app defining routes and orchestrating the pipeline
├── run_server.sh        # helper script to launch the Flask server
├── requirements.txt     # pinned Python dependencies
└── .env.example         # template for environment variables
```

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE.md) for details.

