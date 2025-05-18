from __future__ import annotations
import os
import uuid
import io
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import datetime

# Configure matplotlib before other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'MODEL_FOLDER': 'static/models',
    'ALLOWED_EXTENSIONS': {'jpg', 'jpeg', 'tif', 'tiff', 'h5', 'keras'},
    'MAX_CONTENT_LENGTH': 1024 * 1024 * 1024,  # 1GB
})

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path: str, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess image for model prediction."""
    try:
        if img_path.lower().endswith(('.tif', '.tiff')):
            import tifffile
            img = tifffile.imread(img_path)
            if img.ndim == 2:
                img = np.stack((img,) * 3, axis=-1)
            img = tf.image.resize(img, target_size)
            img = (img / 32767.5) - 1.0  # Normalize 16-bit TIFF
        else:
            img = image.load_img(img_path, target_size=target_size)
            img = image.img_to_array(img)
            img = (img / 127.5) - 1.0  # Normalize 8-bit images
            
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise

def generate_heatmap(model: tf.keras.Model, img_array: np.ndarray, confidence: float) -> str:
    """Generate activation heatmap visualization with result text."""
    try:
        # Find last convolutional layer
        layer_name = next((layer.name for layer in reversed(model.layers) 
                         if isinstance(layer, tf.keras.layers.Conv2D)), None)
        
        if not layer_name:
            raise ValueError("No convolutional layers found in model")

        # Create activation model
        activation_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Generate activations
        activations = activation_model.predict(img_array)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Original image - left subplot
        plt.subplot(1, 2, 1)
        display_img = ((img_array[0] + 1.0) * 127.5).astype('uint8')
        plt.imshow(display_img)
        prediction_label = 'Wildfire' if confidence >= 0.5 else 'No Wildfire'
        plt.title(f"{prediction_label} ({confidence*100:.1f}%)", fontsize=14)
        plt.axis('off')
        
        # Activation map - right subplot
        plt.subplot(1, 2, 2)
        
        # Create a custom colormap similar to the one in your example
        colors = [(0.0, (32/255, 135/255, 130/255)),  # teal
                (0.25, (105/255, 190/255, 40/255)),  # green
                (0.5, (255/255, 255/255, 0/255)),    # yellow
                (0.75, (40/255, 85/255, 180/255)),   # blue
                (1.0, (75/255, 0/255, 130/255))]     # purple
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # Plot the first activation channel with a 3x3 grid
        act_data = activations[0, :, :, 0]
        plt.imshow(act_data, cmap=custom_cmap)
        plt.title(f"Activations: {layer_name}", fontsize=14)
        plt.axis('off')
        
        # Add overall result text at the top of the figure
        plt.suptitle(f"WILDFIRE DETECTION RESULT: {prediction_label.upper()} - Confidence: {confidence*100:.1f}%", 
                     fontsize=16, fontweight='bold', color='red' if confidence >= 0.5 else 'green')
        
        # Make the layout tight but leave space for the title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        
        # Save to file
        filename = f"heatmap_{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(buf.read())
            
        return filename
    except Exception as e:
        logger.error(f"Heatmap generation failed: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(filepath)
        
        # Generate a preview image for immediate display
        preview_filename = f"preview_{save_name}"
        preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
        
        try:
            # Generate a preview image that's easily viewable in the browser
            if filename.lower().endswith(('.tif', '.tiff')):
                # Handle TIFF separately
                import tifffile
                img = tifffile.imread(filepath)
                if img.ndim == 2:  # Grayscale
                    img = np.stack((img,) * 3, axis=-1)
                
                # Normalize and convert to 8-bit
                min_val = np.min(img)
                max_val = np.max(img)
                if max_val > 0:
                    img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                
                # Save as JPEG for preview
                Image.fromarray(img).save(preview_path, format='JPEG', quality=90)
            else:
                # For standard images, just resize and save
                with Image.open(filepath) as img:
                    img = img.copy()  # Create a copy to avoid potential issues
                    img.thumbnail((800, 800))  # Resize while maintaining aspect ratio
                    img.save(preview_path, format='JPEG', quality=90)
        except Exception as e:
            logger.warning(f"Preview generation failed: {str(e)}")
            # If preview fails, we'll continue without it
            preview_filename = None
        
        return jsonify({
            'message': 'File uploaded',
            'filepath': save_name,
            'preview_path': preview_filename
        })
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-model', methods=['POST'])
def upload_model():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['MODEL_FOLDER'], save_name)
        file.save(filepath)
        
        return jsonify({
            'message': 'Model uploaded',
            'model_path': save_name
        })
    except Exception as e:
        logger.error(f"Model upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-model-url', methods=['POST'])
def upload_model_url():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'Missing URL parameter'}), 400
            
        url = data['url']
        if not url:
            return jsonify({'error': 'Invalid URL'}), 400
        
        # Here you would implement URL download logic
        # For security and scope reasons, this is just a placeholder
        # In a real implementation, you'd download from the URL
        
        return jsonify({
            'error': 'Model URL download not implemented',
            'message': 'Feature coming soon'
        }), 501
    except Exception as e:
        logger.error(f"Model URL error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run-model', methods=['POST'])
def run_model():
    try:
        data = request.get_json()
        if not data or 'image_path' not in data or 'model_path' not in data:
            return jsonify({'error': 'Missing parameters'}), 400

        # Validate paths
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], data['image_path'])
        model_path = os.path.join(app.config['MODEL_FOLDER'], data['model_path'])
        
        if not os.path.exists(img_path) or not os.path.exists(model_path):
            return jsonify({'error': 'Invalid file paths'}), 400

        # Load model
        model = load_model(model_path, compile=False)
        
        # Process image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        pred = model.predict(img_array)
        confidence = float(pred[0][0]) if model.output_shape[-1] == 1 else float(np.max(pred))
        prediction = 1 if confidence >= 0.5 else 0

        # Generate result image
        result_filename = f"result_{uuid.uuid4().hex}.png"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        
        # Create visualization image
        try:
            if img_path.lower().endswith(('.tif', '.tiff')):
                import tifffile
                display_img = tifffile.imread(img_path)
                if display_img.ndim == 2:
                    display_img = np.stack((display_img,) * 3, axis=-1)
                min_val = np.min(display_img)
                max_val = np.max(display_img)
                if max_val > min_val:
                    display_img = ((display_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                display_img = np.array(Image.open(img_path))
            
            # Create annotated image
            plt.figure(figsize=(10, 6))
            plt.imshow(display_img)
            plt.title(f"Prediction: {'Wildfire' if prediction else 'No Wildfire'}\nConfidence: {confidence*100:.1f}%", 
                     fontsize=12, pad=20)
            plt.axis('off')
            
            # Save visualization
            plt.savefig(result_filepath, bbox_inches='tight', dpi=150)
            plt.close()
            
        except Exception as e:
            logger.error(f"Result image generation failed: {str(e)}")
            result_filename = None

        # Prepare text result data
        text_result = {
            'prediction': 'Wildfire detected' if prediction else 'No wildfire detected',
            'confidence': f"{confidence*100:.1f}%",
            'model_used': os.path.basename(model_path),
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Properly referenced
            'image_dimensions': dict(zip(['width', 'height'], display_img.shape[:2][::-1]))
        }

        return jsonify({
            'prediction': 'wildfire' if prediction else 'no_wildfire',
            'confidence': round(confidence, 4),
            'result_url': f"/results/{result_filename}" if result_filename else None,
            'text_result': text_result
        })

    except Exception as e:
        logger.error(f"Model execution error: {str(e)}")
        return jsonify({
            'error': str(e),
            'prediction': 'error',
            'confidence': 0
        }), 500

@app.route('/get-text-result')
def get_text_result():
    return jsonify(analysisResult['text_result']), 200   

@app.route('/results/<filename>')
def get_result(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/download-result/<filename>')
def download_result(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)