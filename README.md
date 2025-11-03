🧠 Background Remover Web App

A powerful Flask-based background remover that can process both local image uploads and images from URLs using multiple AI models (U²-Net, U²-Netp, U²-Net Human Segmentation, Silueta).
Built for speed, flexibility, and clean results — all in a simple web interface.

🚀 Features

🖼️ Supports Local & URL Images — Upload from your device or paste an image link.

⚙️ Multiple AI Models — Choose between U²-Net, U²-Netp, Human Segmentation, and Silueta.

💡 Image Enhancement Options — Optional contrast and sharpness improvements before processing.

🧵 Asynchronous Processing — Background jobs handled with a thread pool for efficient performance.

🧹 Automatic Cleanup — Old uploads, results, and temporary files are automatically removed.

🔧 Model Caching System — Keeps downloaded AI models in a cache for faster repeated use.

🧑‍💻 Flask + BackgroundRemover CLI Integration — Combines the backgroundremover command-line tool with a user-friendly web interface.

🧩 Tech Stack

Backend: Flask (Python)

AI Models: U²-Net, U²-Netp, U²-Net Human Segmentation, Silueta

Image Handling: Pillow (PIL)

Async Execution: concurrent.futures + threading

HTTP Requests: Requests library

Frontend: Jinja2 templates (index, processing, result)

📦 Installation
# Clone the repository
git clone https://github.com/yourusername/background-remover-web.git
cd background-remover-web

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py


Then open your browser and visit:

http://127.0.0.1:5000

🖥️ Usage

Upload an image or paste an image URL.

Choose a background removal model.

Optionally enable contrast/sharpness enhancement.

Wait for the background to be removed automatically.

Download your clean, transparent image.