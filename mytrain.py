from ultralytics import YOLO
 

    
# model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
#model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch
model = YOLO("ultralytics/cfg/models/v8/subtleyolo.yaml")
# model = YOLO('yolov8n.pt')
# Train the model

results = model.train(data="visdrone.yaml", epochs=200, imgsz=640, batch=8, verbose=True)





