<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.jpg"
      >
    </a>
  </p>
</div>

# Autodistill: YOLOv11 Target Model

This repository contains the code implementing [YOLOv11](https://github.com/ultralytics/ultralytics) as a Target Model for use with [`autodistill`](https://github.com/autodistill/autodistill). You can also use a YOLOv11 model as a base model to auto-label data. 

YOLOv11 is a Convolutional Neural Network (CNN) that supports realtime object detection, instance segmentation, keypoint detection, and more.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

## Installation

You can install the Autodistill YOLOv11 Target Model with pip:

```bash
pip3 install autodistill-yolov11
```

You will also need to install a base model like Grounded SAM (`autodistill-grounded-sam`) to label data.

You can find a full list of `detection` Base Models on [the main autodistill repo](https://github.com/autodistill/autodistill).

## Quickstart (Train a YOLOv11 Model)

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov11 import YOLOv11

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundedSAM(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)

target_model = YOLOv11("yolo11n.pt")
target_model.train("./dataset/data.yaml", epochs=200)

# run inference on the new model
pred = target_model.predict("./dataset/valid/your-image.jpg", confidence=0.5)
print(pred)

# optional: upload your model to Roboflow for deployment
from roboflow import Roboflow

rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("PROJECT_ID")
project.version(DATASET_VERSION).deploy(model_type="yolov11", model_path=f"./runs/detect/train/")
```

## Quickstart (Use a YOLOv11 Model to Label Data)

```python
from autodistill_yolov11 import YOLOv11Base
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our YOLOv11 classes
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model

# replace weights_path with the path to your YOLOv11 weights file
base_model = YOLOv11Base(ontology=CaptionOntology({"car": "car"}), weights_path="yolo11n.pt")

# run inference on a single image
results = base_model.predict("container.jpeg")

base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)
```

## Choosing a Task

YOLOv11 supports training both object detection and instance segmentation tasks at various sizes (larger models are slower but can be more accurate). This selection is done in the constructor.

For example:
```python
# initializes a nano-sized instance segmentation model
target_model = YOLOv11("yolov11n-seg.pt")
```

Available object detection initialization options are:

* `yolo11n.pt` - nano
* `yolo11s.pt` - small
* `yolo11m.pt` - medium
* `yolo11l.pt` - large
* `yolo11x.pt` - extra-large

## License

The code in this repository is licensed under an [AGPL 3.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!