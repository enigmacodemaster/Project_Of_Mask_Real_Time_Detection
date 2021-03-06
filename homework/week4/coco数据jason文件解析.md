## COCO数据集
### json文件结构
> 

##### captions_val2014.json截取示例
```json
{
    "info" : {"description": "COCO 2014 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2014,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
    }, 
    
    "image" : [
        {"license": 3,
        "file_name": "COCO_val2014_000000391895.jpg",
        "coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg",
        "height": 360,
        "width": 640,
        "date_captured": "2013-11-14 11:18:45",
        "flickr_url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
        "id": 391895},
        
        {"license": 4,
        "file_name": "COCO_val2014_000000522418.jpg",
        "coco_url": "http://images.cocodataset.org/val2014/COCO_val2014_000000522418.jpg",
        "height": 480,
        "width": 640,
        "date_captured": "2013-11-14 11:38:44",
        "flickr_url": "http://farm1.staticflickr.com/1/127244861_ab0c0381e7_z.jpg",
        "id": 522418}
    ],
    
    "licenses" : [
        {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},
        {"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},
        {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},
        {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},
        {"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},
        {"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},
        {"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},
        {"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}
    ],
    
    "annotations" : [
        {"image_id": 203564,"id": 37,"caption": "A bicycle replica with a clock as the front wheel."},
        {"image_id": 179765,"id": 38,"caption": "A black Honda motorcycle parked in front of a garage."}
    ]
}
```


##### instances_val2017.json截取示例
```json
{
    "info": {
    "description": "COCO 2017 Dataset",
    "url": "http://cocodataset.org",
    "version": "1.0",
    "year": 2017,
    "contributor": "COCO Consortium",
    "date_created": "2017/09/01"
    },
    
    "licenses": [
    {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},
    {"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},
    {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},
    {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},
    {"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},
    {"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},
    {"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},
    {"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}
    ],
    
    "images": [
        {"license": 4,
        "file_name": "000000397133.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "height": 427,
        "width": 640,
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133},
        {"license": 1,
        "file_name": "000000037777.jpg",
        "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
        "height": 230,
        "width": 352,
        "date_captured": "2013-11-14 20:55:31",
        "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg",
        "id": 37777}
    ],
    
    "annotations": [
        {"segmentation": [[510.66,423.01,511.72,420.03,510.45,416.0,510.34,413.02,510.77,410.26,510.77,407.5,510.34,405.16,511.51,402.83,511.41,400.49,510.24,398.16,509.39,397.31,504.61,399.22,502.17,399.64,500.89,401.66,500.47,402.08,499.09,401.87,495.79,401.98,490.59,401.77,488.79,401.77,485.39,398.58,483.9,397.31,481.56,396.35,478.48,395.93,476.68,396.03,475.4,396.77,473.92,398.79,473.28,399.96,473.49,401.87,474.56,403.47,473.07,405.59,473.39,407.71,476.68,409.41,479.23,409.73,481.56,410.69,480.4,411.85,481.35,414.93,479.86,418.65,477.32,420.03,476.04,422.58,479.02,422.58,480.29,423.01,483.79,419.93,486.66,416.21,490.06,415.57,492.18,416.85,491.65,420.24,492.82,422.9,493.56,424.39,496.43,424.6,498.02,423.01,498.13,421.31,497.07,420.03,497.07,415.15,496.33,414.51,501.1,411.96,502.06,411.32,503.02,415.04,503.33,418.12,501.1,420.24,498.98,421.63,500.47,424.39,505.03,423.32,506.2,421.31,507.69,419.5,506.31,423.32,510.03,423.01,510.45,423.01]],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67],
        "category_id": 18,
        "id": 1768
        },
        
        {"segmentation": [[289.74,443.39,302.29,445.32,308.09,427.94,310.02,416.35,304.23,405.73,300.14,385.01,298.23,359.52,295.04,365.89,282.3,362.71,275.29,358.25,277.2,346.14,280.39,339.13,284.85,339.13,291.22,338.49,293.77,335.95,295.04,326.39,297.59,317.47,289.94,309.82,287.4,288.79,286.12,275.41,284.21,271.59,279.11,276.69,275.93,275.41,272.1,271.59,274.01,267.77,275.93,265.22,277.84,264.58,282.3,251.2,293.77,238.46,307.79,221.25,314.79,211.69,325.63,205.96,338.37,205.32,347.29,205.32,353.03,205.32,361.31,200.23,367.95,202.02,372.27,205.8,382.52,215.51,388.46,225.22,399.25,235.47,399.25,252.74,390.08,247.34,386.84,247.34,388.46,256.52,397.09,268.93,413.28,298.6,421.91,356.87,424.07,391.4,422.99,409.74,420.29,428.63,415.43,433.48,407.88,414.6,405.72,391.94,401.41,404.89,394.39,420.54,391.69,435.64,391.15,447.51,387.38,461.0,384.68,480.0,354.47,477.73,363.1,433.48,370.65,405.43,369.03,394.64,361.48,398.95,355.54,403.81,351.77,403.81,343.68,403.27,339.36,402.19,335.58,404.89,333.42,411.9,332.34,416.76,333.42,425.93,334.5,430.79,336.12,435.64,321.01,464.78,316.16,468.01,307.53,472.33,297.28,472.33,290.26,471.25,285.94,472.33,283.79,464.78,280.01,462.62,284.33,454.53,285.94,453.45,282.71,448.59,288.64,444.27,291.88,443.74]],
        "area": 27718.476299999995,
        "iscrowd": 0,
        "image_id": 61471,
        "bbox": [272.1,200.23,151.97,279.77],
        "category_id": 18,
        "id": 1773},
        
        {"segmentation": {"counts": [272,2,4,4,4,4,2,9,1,2,16,43,143,24,5,8,16,44,141,25,8,5,17,44,140,26,10,2,17,45,129,4,5,27,24,5,1,45,127,38,23,52,125,40,22,53,123,43,20,54,122,46,18,54,121,54,12,53,119,57,11,53,117,59,13,51,117,59,13,51,117,60,11,52,117,60,10,52,118,60,9,53,118,61,8,52,119,62,7,52,119,64,1,2,2,51,120,120,120,101,139,98,142,96,144,93,147,90,150,87,153,85,155,82,158,76,164,66,174,61,179,57,183,54,186,52,188,49,191,47,193,21,8,16,195,20,13,8,199,18,222,17,223,16,224,16,224,15,225,15,225,15,225,15,225,15,225,15,225,15,225,15,225,15,225,14,226,14,226,14,39,1,186,14,39,3,184,14,39,4,183,13,40,6,181,14,39,7,180,14,39,9,178,14,39,10,177,14,39,11,176,14,38,14,174,14,36,19,171,15,33,32,160,16,30,35,159,18,26,38,158,19,23,41,157,20,19,45,156,21,15,48,156,22,10,53,155,23,9,54,154,23,8,55,154,24,7,56,153,24,6,57,153,25,5,57,153,25,5,58,152,25,4,59,152,26,3,59,152,26,3,59,152,27,1,60,152,27,1,60,152,86,154,80,160,79,161,42,8,29,161,41,11,22,2,3,161,40,13,18,5,3,161,40,15,2,5,8,7,2,161,40,24,6,170,35,30,4,171,34,206,34,41,1,164,34,39,3,164,34,37,5,164,34,35,10,161,36,1,3,28,17,155,41,27,16,156,41,26,17,156,41,26,16,157,27,4,10,25,16,158,27,6,8,11,2,12,6,2,7,159,27,7,14,3,4,19,6,160,26,8,22,18,5,161,26,8,22,18,4,162,26,8,23,15,4,164,23,11,23,11,7,165,19,17,22,9,6,167,19,22,18,8,3,170,18,25,16,7,1,173,17,28,15,180,17,30,12,181,16,34,6,184,15,225,14,226,13,227,12,228,11,229,10,230,9,231,9,231,9,231,9,231,8,232,8,232,8,232,8,232,8,232,8,232,7,233,7,233,7,233,7,233,8,232,8,232,8,232,9,231,9,231,9,231,10,230,10,230,11,229,13,227,14,226,16,224,17,223,19,221,23,217,31,3,5,201,39,201,39,201,39,201,39,201,39,201,40,200,40,200,41,199,41,199,41,199,22,8,12,198,22,12,8,198,22,14,6,198,22,15,6,197,22,16,5,197,22,17,5,196,22,18,4,196,22,19,4,195,22,19,5,194,22,20,4,194,25,21,1,193,27,213,29,211,30,210,35,6,6,193,49,191,50,190,50,190,51,189,51,189,52,188,53,187,53,187,54,186,54,186,54,186,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,55,185,28,1,26,185,23,11,21,185,20,17,17,186,18,21,15,186,16,23,14,187,14,25,14,187,14,26,12,188,14,28,10,188,14,226,14,226,16,224,17,223,19,221,20,220,22,218,24,18,3,12,3,180,25,10,1,4,6,10,6,178,28,7,12,8,8,177,49,3,12,176,65,175,67,173,69,171,53,3,14,170,37,20,9,4,1,169,36,21,8,175,35,22,7,176,34,23,7,176,34,23,6,177,35,22,6,177,35,22,8,175,35,23,9,173,35,205,36,204,39,201,43,197,48,36,1,155,48,35,3,154,49,33,5,154,48,32,6,155,49,27,10,155,51,24,11,154,54,21,11,155,56,19,11,155,56,18,11,156,56,17,11,157,56,16,12,157,56,14,13,159,56,12,13,160,61,5,14,162,78,165,75,167,73,168,72,170,70,171,69,173,67,176,64,179,61,182,58,183,57,185,54,187,53,188,51,191,49,192,47,195,45,196,43,198,42,199,40,201,38,203,36,205,34,207,32,210,28,213,26,216,22,221,16,228,8,10250],"size": [240,320]},
        "area": 18419,
        "iscrowd": 1,
        "image_id": 448263,
        "bbox": [1,0,276,122],
        "category_id": 1,
        "id": 900100448263}
    ],
    
    "categories": [
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
    {"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]
    
}
```