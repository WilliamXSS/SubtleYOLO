



from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model

metrics = model.val(data="coco128.yaml")  # evaluate model performance on the validation set
