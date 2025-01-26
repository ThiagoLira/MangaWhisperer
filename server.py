import io
import base64
import secrets

import numpy as np
from flask import Flask, request, redirect, url_for
from flask import render_template_string
from PIL import Image
from scipy.io import wavfile


app = Flask(__name__)

# A simple in-memory store for results. { key: [ (img_bytes, wav_bytes), ... ] }
# In real apps, use a database or session, not a global dict!
stored_results = {}



def process_image_bak(pil_img):
    """
    Return a list of tuples: [(PIL.Image, np.array), ...].
    For demo, we create 3 dummy (image, audio) pairs.
    """
    pil_img = pil_img.convert("RGB")
    results = []
    
    # Example: Create 3 random-signal audios for the same image
    for _ in range(3):
        duration = 1.0  # seconds
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate*duration), endpoint=False)
        freq = 440  # A4
        audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.05*np.random.randn(len(t))
        results.append((pil_img, audio))
    
    return results

@app.route("/", methods=["GET"])
def index():
    """
    Show a page with a drag-and-drop upload zone and a file input.
    """
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image + Audio Demo</title>
        <style>
          .drop-zone {
              width: 300px;
              height: 100px;
              border: 2px dashed #ccc;
              border-radius: 10px;
              display: flex;
              align-items: center;
              justify-content: center;
              font-family: sans-serif;
              color: #999;
              margin-bottom: 10px;
          }
          .drop-zone.dragover {
              border-color: #333;
              color: #333;
          }
        </style>
    </head>
    <body>
      <h1>Upload an Image</h1>
      <div class="drop-zone" id="dropZone">
          Drop your image here
      </div>
      <form action="/upload" method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" id="fileInput">
        <button type="submit">Upload</button>
      </form>
      
      <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        // Highlight drop zone when file is dragged over
        dropZone.addEventListener('dragover', (e) => {
          e.preventDefault();
          dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', (e) => {
          e.preventDefault();
          dropZone.classList.remove('dragover');
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
          e.preventDefault();
          dropZone.classList.remove('dragover');
          if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
          }
        });
      </script>
    </body>
    </html>
    """)

@app.route("/upload", methods=["POST"])
def handle_upload():
    from image_processor import process_image
    """
    1) Read uploaded image
    2) Process the image
    3) Store the results in memory
    4) Redirect to /results/<unique_key>
    """
    if "image" not in request.files:
        return "No file uploaded!", 400
    file = request.files["image"]
    if not file.filename:
        return "Empty file!", 400
    
    pil_img = Image.open(file.stream).convert("L").convert("RGB")
    pil_img_array = np.array(pil_img)
    # Process the image
    results = process_image(pil_img_array)
    # Convert each (PIL.Image, np.array) to raw bytes
    # so we can store them in memory without re-running the processing later.
    results_bytes = []
    for (img_obj, audio_array) in results:
        # Convert image to PNG bytes
        img_buffer = io.BytesIO()
        img_obj.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()
        
        # Convert audio array to WAV bytes
        wav_buffer = io.BytesIO()
        sample_rate = 22050
        # Scale float array to int16
        scaled = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        wavfile.write(wav_buffer, sample_rate, scaled)
        wav_bytes = wav_buffer.getvalue()
        
        results_bytes.append((img_bytes, wav_bytes))
    
    # Store in global dictionary with a random key
    key = secrets.token_hex(16)
    stored_results[key] = results_bytes
    
    # Redirect to results page
    return redirect(url_for("show_results", result_key=key))

@app.route("/results/<result_key>")
def show_results(result_key):
    """
    Retrieve the stored results from memory and render them.
    """
    if result_key not in stored_results:
        return "Invalid or expired results key.", 404
    
    results_bytes = stored_results[result_key]
    
    # Build the HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Image + Audio Output</title>
</head>
<body>
    <h1>Results</h1>
    <table style="border-collapse: collapse;">
        <tr>
            <th style="border: 1px solid #ccc; padding: 8px;">Image</th>
            <th style="border: 1px solid #ccc; padding: 8px;">Audio</th>
        </tr>
    """
    
    for (img_bytes, wav_bytes) in results_bytes:
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        wav_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        html_content += f"""
        <tr>
            <td style="border: 1px solid #ccc; padding: 8px;">
                <img src="data:image/png;base64,{img_b64}" width="200"/>
            </td>
            <td style="border: 1px solid #ccc; padding: 8px;">
                <audio controls>
                    <source src="data:audio/wav;base64,{wav_b64}" type="audio/wav">
                </audio>
            </td>
        </tr>
        """
    
    html_content += f"""
    </table>
    <p><a href="/">Upload another image</a></p>
</body>
</html>
"""
    return html_content

if __name__ == "__main__":
    # Warning: "debug=True" is not for production
    app.run(debug=True)

