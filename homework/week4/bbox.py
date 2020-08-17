import cv2

img = cv2.imread('000000289343.jpg')
annotation = {"bbox": [473.07,395.93,38.65,28.67], "category_id": 18}
categories =  [
    {"supercategory": "person","id": 1,"name": "person"},
    {"supercategory": "vehicle","id": 2,"name": "bicycle"},
    {"supercategory": "vehicle","id": 3,"name": "car"},
    {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
    {"supercategory": "vehicle","id": 5,"name": "airplane"},
    {"supercategory": "vehicle","id": 6,"name": "bus"},
    {"supercategory": "vehicle","id": 7,"name": "train"},
    {"supercategory": "vehicle","id": 8,"name": "truck"},
    {"supercategory": "vehicle","id": 9,"name": "boat"},
    {"supercategory": "outdoor","id": 10,"name": "traffic light"},
    {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
    {"supercategory": "outdoor","id": 13,"name": "stop sign"},
    {"supercategory": "outdoor","id": 14,"name": "parking meter"},
    {"supercategory": "outdoor","id": 15,"name": "bench"},
    {"supercategory": "animal","id": 16,"name": "bird"},
    {"supercategory": "animal","id": 17,"name": "cat"},
    {"supercategory": "animal","id": 18,"name": "dog"},
]


def drawBbox(img, annotation, categories):
	x_min, y_min, w, h = annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][2], annotation['bbox'][3]
	x_max = x_min + w
	y_max = y_min + h
	cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255, 0), 2)
	font = cv2.FONT_HERSHEY_SIMPLEX
	catagory = annotation["category_id"]
	name = categories[catagory - 2]['name']
	cv2.putText(img, name, (int(x_min - 10), int(y_min-10)), font, 0.5, (0,0,255), 2)

drawBbox(img, annotation, categories)

cv2.imwrite('000000289343_anno.jpg', img)

cv2.imshow('dog', img)

cv2.waitKey(0)
