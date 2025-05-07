from __future__ import annotations
import os
import uuid
import json
import exifread
import torch
import torch.nn as nn
import nbformat
from nbconvert import PythonExporter
from func_timeout import func_timeout, FunctionTimedOut
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
from torchvision import transforms
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from typing import (
    TYPE_CHECKING, Any, Tuple, cast, Union,
    Sequence, Dict, Optional
)
import numpy.typing as npt


if TYPE_CHECKING:
    from tensorflow.keras.models import Model
    from tensorflow.python.types.core import TensorLike
    from PIL.Image import Image as PILImage
else:
    Model = Any
    TensorLike = Any
    PILImage = Any


ResponseReturnValue = Union[
    Response,
    Tuple[Response, int],
    str,
    Tuple[str, int],
    Dict[str, Any],
    Tuple[Dict[str, Any], int]
]


tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Keras imports
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

class WildfireBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

app = Flask(__name__, static_folder='static', template_folder='templates')

app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'MODEL_FOLDER': 'static/models',
    'ALLOWED_EXTENSIONS': {
        'png', 'jpg', 'jpeg', 'tif', 'tiff',
        'pth', 'pt', 'ipynb', 'keras', 'h5'
    },
    'MAX_CONTENT_LENGTH': 1024 * 1024 * 1024,
    'NOTEBOOK_TIMEOUT': 30
})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_tiff_to_png(tiff_path: str) -> str:
    png_path = tiff_path.rsplit('.', 1)[0] + '.png'
    with Image.open(tiff_path) as img:
        img.save(png_path, 'PNG')
    return png_path

def get_gps_coordinates(image_path: str) -> Tuple[float, float]:
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        gps_lat = tags.get('GPS GPSLatitude')
        gps_lon = tags.get('GPS GPSLongitude')
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_ref = tags.get('GPS GPSLongitudeRef')
        
        if all([gps_lat, gps_lon, lat_ref, lon_ref]):
            try:
                lat = cast(list, gps_lat.values)
                lon = cast(list, gps_lon.values)
                
                lat_decimal = float(lat[0]) + float(lat[1]/60) + float(lat[2]/3600)
                lon_decimal = float(lon[0]) + float(lon[1]/60) + float(lon[2]/3600)
                
                if str(lat_ref) != 'N':
                    lat_decimal *= -1
                if str(lon_ref) != 'E':
                    lon_decimal *= -1
                
                return (round(lat_decimal, 6), round(lon_decimal, 6))
            except Exception:
                pass
    return (0.0, 0.0)

def create_heatmap(
    original_img: Image.Image,
    prediction: int,
    confidence: float,
    output_path: str
) -> None:
    base_img = original_img.convert('RGBA')
    overlay = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
    if prediction == 1:
        alpha = int(confidence * 255)
        overlay = Image.new('RGBA', base_img.size, (255, 0, 0, alpha))
    result = Image.alpha_composite(base_img, overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    text = f"{'Wildfire Detected' if prediction == 1 else 'No Wildfire'} ({confidence*100:.1f}%)"
    draw.text((10, 10), text, fill=(255, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    result.save(output_path)

def convert_ipynb_to_model(ipynb_path: str) -> str:
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, "model.pth")
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(notebook)
        global_namespace = {
            '__builtins__': __builtins__,
            'torch': torch,
            'nn': nn,
            'WildfireBaseModel': WildfireBaseModel,
            'temp_model_path': temp_model_path
        }
        exec(script, global_namespace)
        if not os.path.exists(temp_model_path):
            raise RuntimeError("No valid model file created by notebook")
        final_path = os.path.join(app.config['MODEL_FOLDER'], f"model_{uuid.uuid4().hex}.pth")
        shutil.move(temp_model_path, final_path)
        return final_path
    except Exception as e:
        raise RuntimeError(f"Notebook processing failed: {str(e)}") from e
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def prepare_image(
    img: Image.Image,
    model_input_shape: Tuple[int, ...]
) -> npt.NDArray[np.float32]:
    
    if len(model_input_shape) == 4:
        _, height, width, channels = model_input_shape
    else:
        height, width, channels = model_input_shape

    
    if channels == 1:
        img_resized = img.resize((width, height)).convert('L')
    else:
        img_resized = img.resize((width, height)).convert('RGB')

    img_array = np.array(img_resized).astype('float32')
    
    
    if channels == 3:
        img_array = preprocess_input(img_array)      
    else:
        img_array = (img_array / 127.5) - 1.0  

    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index() -> str:
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image() -> ResponseReturnValue:
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename or '')
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file type'}), 400

        ext = filename.split('.')[-1].lower()
        new_filename = f"img_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(filepath)
        
        if ext in {'tif', 'tiff'}:
            png_path = convert_tiff_to_png(filepath)
            return jsonify({
                'message': 'TIFF uploaded and converted',
                'filepath': os.path.basename(png_path)
            })
        return jsonify({
            'message': 'Image uploaded successfully',
            'filepath': new_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload-model', methods=['POST'])
def upload_model() -> ResponseReturnValue:
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename or '')
        if not allowed_file(filename):
            return jsonify({'error': 'Invalid file format'}), 400

        ext = filename.split('.')[-1]
        new_filename = f"model_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['MODEL_FOLDER'], new_filename)
        file.save(filepath)
        return jsonify({
            'message': 'File uploaded successfully',
            'model_path': new_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run-model', methods=['POST'])
def run_model() -> ResponseReturnValue:
    try:
        data = request.get_json()
        if not data or 'image_path' not in data or 'model_path' not in data:
            return jsonify({'error': 'Missing parameters'}), 400

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], str(data['image_path']))
        model_path = os.path.join(app.config['MODEL_FOLDER'], str(data['model_path']))

        if model_path.endswith('.ipynb'):
            try:
                model_path = func_timeout(
                    app.config['NOTEBOOK_TIMEOUT'],
                    convert_ipynb_to_model,
                    args=(model_path,)
                )
            except FunctionTimedOut:
                return jsonify({'error': 'Notebook execution timed out'}), 400

        if not os.path.exists(img_path) or not os.path.exists(model_path):
            return jsonify({'error': 'Invalid file paths'}), 400

        img = Image.open(img_path).convert('RGB')
        result_base = f"result_{uuid.uuid4().hex}"
        json_filename = f"{result_base}.json"
        png_filename = f"{result_base}.png"
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        png_path = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)

        if model_path.endswith(('.keras', '.h5')):
            try:
                model = load_model(model_path, compile=False)
                input_shape = cast(Tuple[int, ...], model.input_shape)
                img_array = prepare_image(img, input_shape)
                
                prediction = model.predict(img_array)
                
                # Handle different output types
                if prediction.shape[-1] == 1:  # Sigmoid output
                    confidence = float(prediction[0][0])
                    pred_class = 1 if confidence >= 0.5 else 0
                else:  # Softmax output
                    confidence = float(np.max(prediction))
                    pred_class = int(np.argmax(prediction))
            except Exception as e:
                return jsonify({'error': f'Keras model error: {str(e)}'}), 500
        elif model_path.endswith(('.pth', '.pt')):
            try:
                model = WildfireBaseModel()
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                img_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence_tensor, pred_class_tensor = torch.max(probabilities, 0)
                    confidence = confidence_tensor.item()
                    pred_class = pred_class_tensor.item()
            except Exception as e:
                return jsonify({'error': f'PyTorch model error: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Unsupported model format'}), 400

        json.dump({
            "prediction": "wildfire_detected" if pred_class == 1 else "no_wildfire",
            "confidence": round(confidence, 4),
            "coordinates": get_gps_coordinates(img_path)
        }, open(json_path, 'w'))

        create_heatmap(img, pred_class, confidence, png_path)

        return jsonify({
            'message': 'Processing complete',
            'result_path': png_filename,
            'json_path': json_filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-result/<filename>')
def download_result(filename: str) -> ResponseReturnValue:
    try:
        safe_filename = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Server failed to start: {str(e)}")
        raise
    