
# This is a test for maingames

## Description
### Pipeline
Hero Avatar Detection (1) -> Classifier Embedding (2) -> Cosine Similarity Retrieval (3)

(1) Hero Avatar Detection: Detect hero avatar in the image and crop it out. The cropped image will be used as input for the classifier embedding.
I use opencv2 to detect blob in the image, if not found, the image will be cropped as rectangle.

(2) Classifier Embedding: Classify the hero avatar into one of the 107 classes (in folder "hero_images"). The choosen backbone is VGG19.
This model trained on 107 classes with 100 accuracy on training dataset and 98 accuracy on public test dataset.

(3) Cosine Similarity Retrieval: Retrieve the most similar hero avatar from the database (in folder "hero_images") based on the classifier embedding. Current accuracy is 98% on public test dataset.

### Remaining problems
- The Hero Avatar Detection currently is a simple blob detector, it can be improved by using a object detection model.
- The Cosine Similarity Retrieval can be improved by using a faster algorithm like [FAISS](https://github.com/facebookresearch/faiss)

## Install
```
pip install --upgrade pip
pip install -r requirements.txt
```

## use docker (cuda 11.2)
```
docker build -t maingames:latest .
docker run -it --name test_maingames --gpus all -v $(pwd):/src maingames:latest
``` 

### Evaluate detector (export outputs to test_data/test_images_processed)
```
python3 extract_detector.py
```

## Train the classifier
```
python3 train.py
```

## Evaluate the pretrained classifier
```
python3 test.py
```

## Predict by retrieving
```
python3 retrieve.py --data_path test_data/test_images --output_path test.txt
```