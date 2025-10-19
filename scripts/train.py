from ultralytics import YOLO

def train_model():
    """Train YOLO model"""
    model = YOLO('yolov8n.pt')
    results = model.train(data='config.yaml', epochs=50)
    return results
