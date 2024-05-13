from ultralytics import YOLO
 


model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  # build a new model from scratch

 
# Train the model
results = model.train(data="coco128.yaml", epochs=2000, imgsz=1280,batch=18)

