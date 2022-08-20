

https://user-images.githubusercontent.com/56389219/185725238-bcf8a72f-a86d-4d8e-b0ec-b52561d01676.mp4


# Intrusion-Detection-System-IDS-

1. Dataset

  - Domain: person
  - Source: three video yotube with person walking on street (30 fps, total ~85 minute)
  - Pre-processing data:
    + Extract image in three video (Skip 30 frame, final: ~5000 image)
    + Labeling per image using lib LabelImg
    + Split data to train model (train & val)
    + Test data: ~500 image from Kaggle (follow: https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset)

2. Models:
  - Using yolov5 custom
  - Train model, transfer learning on yolov5s  
  - Export model to ONNX format & inference by opencv (function ReadNetFromONNX)

3. Pipeline IDS:
  - Create coords of polygon (Area need to detection)
  - In detect, find centroid per object
  - Check the centroid with area
  - While true: Send massage to telegram user
  
 4. Interface:
 - Deverlop with streamlit (Open lib)
 - Home
 ![295537838_474927911144079_1895881611765457322_n](https://user-images.githubusercontent.com/56389219/184522385-dacc8225-677b-48df-9a2a-3158817dbe5e.png)
 - Cam 1
 ![295807255_636727374749088_6899426403236069256_n](https://user-images.githubusercontent.com/56389219/184522391-551e65fd-f24f-4571-a0cf-eeaf91e5bd8e.png)
 - Cam 2
 ![297418638_625767509118855_5531022389000749241_n](https://user-images.githubusercontent.com/56389219/184522397-79422244-9a86-4ce5-937a-94cc55246172.png)
 - Start detection & result
 ![293242546_590862862676432_213999326877175280_n](https://user-images.githubusercontent.com/56389219/184522405-357f8a79-77bf-438d-8f69-05b0ad5dfbce.png)
 - Message on telegram
 ![293770516_445874467270448_7703386803049832963_n](https://user-images.githubusercontent.com/56389219/184522418-b1358f31-ba55-488e-9ce8-cc89e9af10b4.png)


THANKS FOR WATCHING!
 
