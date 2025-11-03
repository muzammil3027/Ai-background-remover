# app.py
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import os
import uuid
import shutil
import subprocess
import requests
from urllib.parse import urlparse
from PIL import Image, ImageEnhance
import threading
import time
import concurrent.futures
import tempfile
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size
app.config['CACHE_FOLDER'] = 'model_cache'

# Create necessary folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER'], 
               app.config['TEMP_FOLDER'], app.config['CACHE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global variables for optimization
model_timestamps = {}  # Track when models were last used
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)  # Limit concurrent background processing
active_jobs = {}  # Track active processing jobs
processing_lock = threading.Lock()  # Lock for thread safety

# Define the models and their settings
MODELS = {
    'u2net': {'description': 'General purpose, good balance of quality and speed'},
    'u2netp': {'description': 'Faster but less accurate model'},
    'u2net_human_seg': {'description': 'Specialized for portraits and people'},
    'silueta': {'description': 'Alternative model for complex backgrounds'}
}


@app.route('/')
def index():
    return render_template('index.html', models=MODELS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        original_ext = os.path.splitext(file.filename)[1].lower()
        if not original_ext:
            original_ext = '.jpg'  # Default extension
        
        filename = unique_id + original_ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get processing parameters from form
        model = request.form.get('method', 'u2net')
        
        # Pre-processing parameters
        enhance_contrast = request.form.get('enhance_contrast', 'false') == 'true'
        enhance_sharpness = request.form.get('enhance_sharpness', 'false') == 'true'
        
        if enhance_contrast or enhance_sharpness:
            preprocess_image(filepath, enhance_contrast, enhance_sharpness)
        
        # Start background processing with progress tracking
        executor.submit(process_image_async, filepath, unique_id, model)
        
        # Return immediately with a waiting page
        return render_template('processing.html', 
                               job_id=unique_id,
                               original=url_for('uploaded_file', filename=filename))

@app.route('/process_url', methods=['POST'])
def process_url():
    if 'image_url' not in request.form or not request.form['image_url'].strip():
        return redirect(url_for('index'))
    
    image_url = request.form['image_url'].strip()
    
    # Validate URL
    try:
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL")
    except:
        return render_template('index.html', 
                              error="Invalid URL provided. Please enter a valid image URL.",
                              models=MODELS)
    
    try:
        # Generate unique ID for this processing job
        unique_id = str(uuid.uuid4())
        
        # Download the image
        response = requests.get(image_url, stream=True, timeout=15)
        
        # Check if the request was successful
        if response.status_code != 200:
            return render_template('index.html', 
                                  error=f"Failed to download image. Status code: {response.status_code}",
                                  models=MODELS)
        
        # Check content type to ensure it's an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            return render_template('index.html', 
                                  error="The URL does not point to a valid image.",
                                  models=MODELS)
        
        # Extract extension from URL or content-type
        if 'image/jpeg' in content_type or 'image/jpg' in content_type:
            extension = '.jpg'
        elif 'image/png' in content_type:
            extension = '.png'
        elif 'image/gif' in content_type:
            extension = '.gif'
        elif 'image/webp' in content_type:
            extension = '.webp'
        else:
            extension = '.jpg'  # Default to jpg
        
        # Save downloaded image
        filename = unique_id + extension
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Get processing parameters from form
        model = request.form.get('method', 'u2net')
        enhance_contrast = request.form.get('enhance_contrast', 'false') == 'true'
        enhance_sharpness = request.form.get('enhance_sharpness', 'false') == 'true'
        
        if enhance_contrast or enhance_sharpness:
            preprocess_image(filepath, enhance_contrast, enhance_sharpness)
        
        # Start background processing
        executor.submit(process_image_async, filepath, unique_id, model)
        
        # Return with a waiting page
        return render_template('processing.html', 
                               job_id=unique_id,
                               original=url_for('uploaded_file', filename=filename))
                              
    except requests.exceptions.RequestException as e:
        return render_template('index.html', 
                              error=f"Error downloading image: {str(e)}",
                              models=MODELS)
    except Exception as e:
        logger.error(f"URL processing error: {e}", exc_info=True)
        return render_template('index.html', 
                              error=f"An error occurred: {str(e)}",
                              models=MODELS)
                              

@app.route('/check_status/<job_id>')
def check_status(job_id):
    """Check the status of a background removal job"""
    if job_id in active_jobs:
        return jsonify({'status': 'processing'})
    
    # Check if the result file exists
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_result.png")
    if os.path.exists(result_path):
        return jsonify({
            'status': 'complete',
            'redirect': url_for('show_result', job_id=job_id)
        })
    
    # If not processing and no result, it failed
    return jsonify({
        'status': 'failed',
        'redirect': url_for('show_result', job_id=job_id, error='true')
    })

@app.route('/result/<job_id>')
def show_result(job_id):
    """Show the result page for a completed job"""
    # Find the original file
    original_file = None
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.startswith(job_id):
            original_file = f
            break
    
    if not original_file:
        return render_template('index.html', 
                              error="Original image not found",
                              models=MODELS)
    
    # Check if result exists
    result_file = f"{job_id}_result.png"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    
    error = request.args.get('error', 'false') == 'true'
    
    return render_template('result.html', 
                          original=url_for('uploaded_file', filename=original_file),
                          result=url_for('result_file', filename=result_file),
                          unique_id=job_id,
                          success=(not error))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename), 
                    as_attachment=True, 
                    download_name='background_removed.png')

def process_image_async(filepath, unique_id, model='u2net'):
    """Process the image in a background thread"""
    with processing_lock:
        active_jobs[unique_id] = time.time()
    
    try:
        # Create output path
        output_filename = f"{unique_id}_result.png"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Process the image
        result = process_with_backgroundremover(filepath, output_path, model)
        
        if not result:
            # If processing failed, create a copy of the original
            shutil.copy(filepath, output_path)
    except Exception as e:
        logger.error(f"Error in async processing: {e}", exc_info=True)
        # Create a copy of the original as fallback
        try:
            output_path = os.path.join(app.config['RESULTS_FOLDER'], f"{unique_id}_result.png")
            shutil.copy(filepath, output_path)
        except:
            pass
    finally:
        # Remove the job from active jobs
        with processing_lock:
            if unique_id in active_jobs:
                del active_jobs[unique_id]
        
        # Clean up temporary files
        cleanup_temp_files()
        
        # Force garbage collection to free memory
        gc.collect()

def preprocess_image(filepath, enhance_contrast=False, enhance_sharpness=False):
    """Apply preprocessing to enhance image quality"""
    try:
        img = Image.open(filepath)
        
        # Resize if image is very large (for better performance)
        max_size = 2000  # maximum dimension
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Apply enhancements if requested
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        if enhance_sharpness:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Increase sharpness by 50%
        
        # Save the preprocessed image
        img.save(filepath)
        return True
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return False

def setup_model_cache(model_name):
    """Ensure the model exists in the cache directory and is symlinked/copied to the .u2net folder"""
    cache_dir = app.config['CACHE_FOLDER']
    home_dir = os.path.expanduser("~")
    u2net_dir = os.path.join(home_dir, ".u2net")
    os.makedirs(u2net_dir, exist_ok=True)
    
    # Update the timestamp for this model
    model_timestamps[model_name] = time.time()
    
    # Check if we need to do anything with the model
    model_file = f"{model_name}.pth"
    cache_path = os.path.join(cache_dir, model_file)
    u2net_path = os.path.join(u2net_dir, model_file)
    
    # If model exists in u2net directory, we're good
    if os.path.exists(u2net_path) and os.path.getsize(u2net_path) > 1000000:
        return True
    
    # If model exists in cache, copy/link it to u2net directory
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000000:
        try:
            # Try symlink first (more efficient)
            if hasattr(os, 'symlink'):
                if os.path.exists(u2net_path):
                    os.remove(u2net_path)
                os.symlink(cache_path, u2net_path)
            else:
                # On Windows without admin, fall back to copy
                shutil.copy(cache_path, u2net_path)
            return True
        except Exception as e:
            logger.warning(f"Could not link/copy model {model_name} from cache: {e}")
    
    # Model not in cache or u2net directory
    return False

def process_with_backgroundremover(input_path, output_path, model='u2net'):
    """Remove background using the backgroundremover CLI tool with optimizations"""
    # Ensure we have a valid model name
    valid_models = ['u2net', 'u2netp', 'u2net_human_seg', 'silueta']
    if model not in valid_models:
        model = 'u2net'
    
    # Prepare model cache
    setup_model_cache(model)
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory(dir=app.config['TEMP_FOLDER']) as temp_dir:
        # Optimize input image if it's large
        optimized_input = os.path.join(temp_dir, "optimized_input.png")
        optimize_input_image(input_path, optimized_input)
        
        # Use the optimized input if it exists
        actual_input = optimized_input if os.path.exists(optimized_input) else input_path
        
        # Build the command with performance optimizations
        cmd = [
            'backgroundremover',
            '-i', actual_input,
            '-o', output_path,
            '-m', model
        ]
        
        # Execute the command with a timeout
        try:
            start_time = time.time()
            logger.info(f"Starting background removal with model {model}")
            
            # Print the command being executed for debugging
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True,
                timeout=120  # 2 minute timeout
            )
            
            # Log stdout and stderr for debugging
            logger.info(f"Command stdout: {result.stdout.decode() if result.stdout else 'No output'}")
            if result.stderr:
                logger.warning(f"Command stderr: {result.stderr.decode()}")
            
            logger.info(f"Background removal completed in {time.time() - start_time:.2f} seconds")
            
            # Check if output was created and is a valid image
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                try:
                    # Verify the output is a valid image
                    with Image.open(output_path) as img:
                        # Log image details for debugging
                        logger.info(f"Output image created: {img.format}, size: {img.size}, mode: {img.mode}")
                    return True
                except Exception as e:
                    logger.error(f"Invalid output image created: {e}")
                    return False
            else:
                logger.error(f"Output file missing or too small: {output_path}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Background removal timed out after 120 seconds")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.stderr.decode() if e.stderr else 'No error output'}")
            return False
        except Exception as e:
            logger.error(f"Error in background removal: {e}")
            return False

def optimize_input_image(input_path, output_path):
    """Optimize the input image for faster processing"""
    try:
        with Image.open(input_path) as img:
            # If image is already small enough, don't bother optimizing
            if max(img.size) <= 1500:
                return False
            
            # Calculate new size while maintaining aspect ratio
            width, height = img.size
            max_size = 1500
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Resize the image using high-quality downsampling
            resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save as PNG for better quality
            resized.save(output_path, format="PNG")
            return True
    except Exception as e:
        logger.error(f"Failed to optimize input image: {e}")
        return False

def cleanup_temp_files():
    """Clean up temporary files that are older than 1 hour"""
    temp_dir = app.config['TEMP_FOLDER']
    current_time = time.time()
    
    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isfile(item_path):
                # If file is older than 1 hour, delete it
                if current_time - os.path.getmtime(item_path) > 3600:
                    os.remove(item_path)
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")

@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    """Admin endpoint to clean up old files"""
    if request.form.get('secret') != 'your_secret_key':
        return jsonify({'error': 'Unauthorized'}), 401
    
    cleanup_uploads()
    cleanup_results()
    cleanup_temp_files()
    
    return jsonify({'success': True})

def cleanup_uploads():
    """Clean up upload files older than 24 hours"""
    cleanup_directory(app.config['UPLOAD_FOLDER'], 24 * 3600)

def cleanup_results():
    """Clean up result files older than 24 hours"""
    cleanup_directory(app.config['RESULTS_FOLDER'], 24 * 3600)

def cleanup_directory(directory, max_age):
    """Clean up files in a directory that are older than max_age seconds"""
    current_time = time.time()
    
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                # If file is older than max_age, delete it
                if current_time - os.path.getmtime(item_path) > max_age:
                    os.remove(item_path)
    except Exception as e:
        logger.error(f"Error cleaning up directory {directory}: {e}")

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('index.html', 
                          error="File too large. Maximum file size is 32MB.",
                          models=MODELS), 413

if __name__ == '__main__':
    # Schedule cleanup on startup
    threading.Thread(target=cleanup_temp_files).start()
    threading.Thread(target=cleanup_uploads).start()
    threading.Thread(target=cleanup_results).start()
    
    app.run(debug=True)