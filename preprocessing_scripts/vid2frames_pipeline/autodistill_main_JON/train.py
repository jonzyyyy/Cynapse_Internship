from autodistill.detection import CaptionOntology
from autodistill_yolov11 import YOLOv11
from autodistill_grounding_dino import GroundingDINO
import os
import csv

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
# names:
#   0: "car"
#   1: "motorcycle"
#   2: "bus"
#   3: "truck"
#   4: "bicycle"
#   5: "scooter"

base_model = GroundingDINO(ontology=CaptionOntology({
              "car": "car",
              "motorcycle": "motorcycle",
              "bus": "bus",
              "truck": "truck",
              "bicycle": "bicycle",
              "scooter": "scooter"
            }))

# base_model = GroundingDINO(ontology=CaptionOntology({
#               "stroller": "stroller",
#               "wheelchair": "wheelchair",
#               "scooter": "scooter",
#               "bicycle": "bicycle"
#             }))


# label all images in a folder called `context_images`
# base_model.label("./data/", extension=".jpg", nms_settings="class_agnostic", iou=0.3, split_ratio=1)

results = base_model.label("./data/", extension=".jpg", nms_settings="class_agnostic", iou=0.7, split_ratio=1)


# Create the CSV file in the ./data directory (optional)

csv_path = os.path.join("./data", "object_confidences.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["object_id", "confidence"])

    for img in results:
        # Unpack: (image_path, image_array, detections_obj)
        image_path, _, detections_obj = img
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        if not hasattr(detections_obj, "confidence"):
            print(f"No confidence found for {image_id}")
            continue

        num_dets = len(detections_obj.confidence)
        for idx in range(num_dets):
            conf = float(detections_obj.confidence[idx])
            obj_id = f"{image_id}_{idx}"
            writer.writerow([obj_id, conf])

print(f"Saved confidences for all objects to {csv_path}")