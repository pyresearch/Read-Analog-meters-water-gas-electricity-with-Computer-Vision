

from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from datetime import datetime
import numpy as np
import pyresearch

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Create folders if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load your custom YOLO model (place best.pt in the project root)
model = YOLO('best.pt')  # Replace with path if needed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path, result_path):
    # Run YOLO inference
    results = model(image_path)[0]
    
    # Plot results on image
    annotated_img = results.plot()  # Draws boxes and labels
    
    # Save annotated image
    cv2.imwrite(result_path, annotated_img)
    
    # Extract detections with positions
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls]
        xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
        x_center = (xyxy[0] + xyxy[2]) / 2
        detections.append({'label': label, 'confidence': round(conf, 2), 'x_center': x_center})
    
    # Sort by x_center (left to right)
    sorted_dets = sorted(detections, key=lambda d: d['x_center'])
    
    # Form meter reading by concatenating labels (assuming digits)
    meter_reading = ''.join([d['label'] for d in sorted_dets]) if sorted_dets else "No detection"
    
    return sorted_dets, meter_reading

def calculate_bill(consumption, utility_type='electricity'):
    if consumption <= 0:
        return 0, "Invalid consumption"
    
    bill_details = ""
    if utility_type == 'electricity':
        #  residential slabs 2025 (Rs/unit, progressive approx)
        slabs = [
            (50, 16.48),   # 0-50
            (50, 20.85),   # 51-100
            (100, 22.94),  # 101-200
            (100, 27.14),  # 201-300
            (400, 32.03),  # 301-700
            (float('inf'), 35.24)  # >700
        ]
        
        energy_charge = 0
        remaining = consumption
        for slab_units, rate in slabs:
            if remaining <= 0:
                break
            consumed = min(remaining, slab_units)
            energy_charge += consumed * rate
            remaining -= consumed
        
        # Fixed charges, taxes (approx)
        fixed = 100
        gst = energy_charge * 0.17
        income_tax = energy_charge * 0.075 if consumption > 200 else 0
        total_bill = energy_charge + fixed + gst + income_tax
        
        bill_details = f"Energy Charge: Rs {energy_charge:.2f}\nFixed: Rs {fixed:.2f}\nGST: Rs {gst:.2f}\nIncome Tax: Rs {income_tax:.2f}\nTotal: Rs {total_bill:.2f}"
        return round(total_bill, 2), bill_details
    
    elif utility_type == 'gas':
        # Placeholder for gas bill (e.g., SSGC rates; add actual slabs)
        rate = 1500 / 100  # Approx Rs15 per unit (m3)
        total_bill = consumption * rate
        bill_details = f"Approx Bill (Rs15/unit): Rs {total_bill:.2f}"
        return round(total_bill, 2), bill_details
    
    elif utility_type == 'water':
        # Placeholder for water bill (e.g., KW&SB rates; add actual)
        rate = 50 / 1000  # Approx Rs0.05 per liter
        total_bill = consumption * rate
        bill_details = f"Approx Bill (Rs0.05/liter): Rs {total_bill:.2f}"
        return round(total_bill, 2), bill_details
    
    return 0, "No bill calc for this type"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!')
            return redirect(request.url)
        
        file = request.files['file']
        utility_type = request.form.get('utility_type', 'electricity')
        
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            
            file.save(upload_path)
            
            detections, meter_reading = process_image(upload_path, result_path)
            
            # Treat current reading as consumption (general, no previous needed)
            try:
                consumption = int(meter_reading) if meter_reading.isdigit() else 0
                bill_amount, bill_details = calculate_bill(consumption, utility_type)
            except ValueError:
                consumption = 0
                bill_amount = 0
                bill_details = "Invalid reading"
            
            # Paths for HTML
            original_rel = os.path.join('uploads', filename)
            result_rel = os.path.join('results', filename)
            
            return render_template('index.html',
                                   original=original_rel,
                                   result=result_rel,
                                   detections=detections,
                                   reading=meter_reading,
                                   consumption=consumption,
                                   bill_amount=bill_amount,
                                   bill_details=bill_details,
                                   utility_type=utility_type)
        else:
            flash('Invalid file type!')
            return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)