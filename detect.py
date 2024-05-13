from ultralytics import YOLO
 
model = YOLO('runs/detect/train71/weights/best.pt')

results = model('../autodl-tmp/datasets/visdrone/VisDrone2019-DET-test-dev/images', save=True)
#results = model('../autodl-tmp/datasets/visdrone/VisDrone2019-DET-test-dev/images/9999938_00000_d_0000203.jpg', save=True)