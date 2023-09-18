import torch

from models.yolo import Model
from models.experimental import attempt_load

model = attempt_load("E:/AIR/yolov5l2/weights/best_ap50.pt", map_location='cpu', fuse=False) # load FP32 model
#print(model)
torch.save(model.state_dict(), "D:/university/code/yolov5/runs/AIR-SARShip/AIR79.pt", _use_new_zipfile_serialization=False)