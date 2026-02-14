import torch
import cv2
import numpy as np
from torchvision.transforms.v2 import ToImage, ToDtype, Resize, Compose, Normalize
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.transforms import functional as F
from torch import nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

# ============= DETECTION CONSTANTS =============
CAR_CLASS = 3 
TRUCK_CLASS = 8
DETECTION_THRESHOLD = 0.5
SIZE_THRESHOLD = 0.005
BORDER_THRESHOLD = 0.1
BLOCKED_THRESHOLD = 0.4

# ============= DRAWING CONSTANTS =============
GREEN = (18, 127, 15)
GRAY = (218, 227, 218)
RED = (220, 20, 60)
line_scale_factor = 0.003
font_scale_factor = 0.0007

# ============= MODEL CLASSES =============
class CNN(nn.Module):
    def __init__(self, classes=10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((32,32))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.AdaptiveMaxPool2d((8,8))
        )  

        self.dense = nn.Sequential(
            nn.Linear(512*8*8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, classes),
            nn.BatchNorm1d(classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        x = self.dense(x)
        return x

    
class CNNTransfer(nn.Module):
    def __init__(self, classes=10):
        super(CNNTransfer, self).__init__()
        
        self.efficientnet = models.efficientnet_v2_s(weights='DEFAULT')
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, classes),
            nn.BatchNorm1d(classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.efficientnet(x)
        return x


# ============= DETECTION FUNCTIONS =============
def is_blocked(box, other_boxes):
    x_min, y_min, x_max, y_max = map(int, box)
    box_area = (x_max - x_min) * (y_max - y_min)
    for other_box in other_boxes:
        ox_min, oy_min, ox_max, oy_max = map(int, other_box)
        
        inter_x_min = max(x_min, ox_min)
        inter_y_min = max(y_min, oy_min)
        inter_x_max = min(x_max, ox_max)
        inter_y_max = min(y_max, oy_max)
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        if inter_area / box_area > BLOCKED_THRESHOLD:  
            return True
        
    return False


def detect_and_crop_cars(input_image):
    cars_image, cars_coords = [], []
    
    model = retinanet_resnet50_fpn_v2(weights='COCO_V1')
    model.to(device)
    model.eval()

    image_tensor = F.to_tensor(input_image).unsqueeze(0)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)
        
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    image_width, image_height = input_image.size
    
    car_boxes = [
        boxes[i].tolist()
        for i in range(len(labels))
        if (labels[i] == CAR_CLASS or labels[i] == TRUCK_CLASS) and scores[i] > DETECTION_THRESHOLD
    ]
    
    filtered_boxes = []
    for box in car_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        box_width = x_max - x_min
        box_height = y_max - y_min

        if (box_width * box_height) / (image_width * image_height) < SIZE_THRESHOLD:
            continue

        if (x_mid / image_width < BORDER_THRESHOLD or 
            y_mid / image_height < BORDER_THRESHOLD or
            x_mid / image_width > 1 - BORDER_THRESHOLD or 
            y_mid / image_height > 1 - BORDER_THRESHOLD):
            continue

        if is_blocked(box, filtered_boxes):
            continue

        filtered_boxes.append(box)

    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        car = input_image.crop((x_min, y_min, x_max, y_max))
        cars_image.append(car)
        cars_coords.append([x_min, y_min, x_max, y_max])
        
    return cars_coords, cars_image


# ============= CLASSIFICATION FUNCTIONS =============
def get_predictions(model, image, transforms):
    img_unsqueezed = transforms(image).unsqueeze(0)
    img_unsqueezed = img_unsqueezed.to(device)
    pred = model(img_unsqueezed)
    probs = torch.exp(pred)
    return probs.cpu()


def ensemble(probs_1, probs_2, method='max'):
    if method == 'product':
        product_probs = probs_1 * probs_2
        product_probs = product_probs.sqrt()
        return product_probs / product_probs.sum()
    elif method == 'arithmatic':
        return (probs_1 + probs_2) / 2
    elif method == 'max':
        product_probs = probs_1 * probs_2
        product_probs = product_probs / product_probs.sum()
        stacked = torch.stack([probs_1, probs_2, product_probs], dim=0)
        max_probs, _ = torch.max(stacked, dim=0)
        return max_probs
    elif method == '1':
        return probs_1
    elif method == '2':
        return probs_2


def predict_car_brand(cars, id_to_class, model1_path='models/model_ft.pt', model2_path='models/model_tl.pt'):
    model1 = CNN(classes=len(id_to_class))
    model2 = CNNTransfer(classes=len(id_to_class))
    
    transforms1 = Compose([
        Resize([96, 128]),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transforms2 = Compose([
        Resize([384, 384]),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.to(device)
    model1.eval()
    
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.to(device)
    model2.eval()
    
    predicted_brands, predicted_probs = [], []
    for img in cars:
        with torch.no_grad():
            probs_1 = get_predictions(model1, img, transforms1)        
            probs_2 = get_predictions(model2, img, transforms2)        
        
        probs = ensemble(probs_1, probs_2, method='max')
        
        label = torch.argmax(probs, dim=1).item()
        predicted_prob = probs[0, label].item()
        predicted_brand = id_to_class[str(label)]  
        
        predicted_brands.append(predicted_brand)
        predicted_probs.append(predicted_prob)
        
    return predicted_brands, predicted_probs


# ============= DRAWING FUNCTIONS =============
def draw_box(image, box):
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    color = GREEN
    
    thickness = max(1, int(line_scale_factor * ((image_width + image_height) / 2)))
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=thickness)
    return image


def draw_label(image, box, text):
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    back_color = GREEN
    text_color = GRAY

    font_scale = font_scale_factor * ((image_width + image_height) / 2)
    thickness = max(1, int(font_scale * 3))
    font = cv2.FONT_HERSHEY_DUPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, thickness)
    
    back_tl = x_min, max(0, y_min - int(1.3 * text_h))
    back_br = x_min + text_w, y_min
    cv2.rectangle(image, back_tl, back_br, back_color, -1)

    text_tl = x_min, max(0, y_min - int(0.2 * text_h))
    cv2.putText(image, text, text_tl, font, font_scale,
                text_color, thickness=thickness, lineType=cv2.LINE_AA)    
    
    return image


def draw_box_and_label(image, cars_bbox, predicted_brands, predicted_probs):
    image = np.array(image)

    for i in reversed(range(len(cars_bbox))):
        image = draw_box(image, cars_bbox[i])

    for i in reversed(range(len(cars_bbox))):
        label = " ".join([predicted_brands[i], str(round(100 * predicted_probs[i], 1)) + "%"])
        image = draw_label(image, cars_bbox[i], label)
    
    return image