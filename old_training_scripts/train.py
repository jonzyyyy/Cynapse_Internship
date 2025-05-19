import mlflow
from mlflow.models.signature import infer_signature
from mlflow.data.sources import LocalArtifactDatasetSource
import os
from ultralytics import YOLO
import yaml

# import onnx
# import onnxruntime as ort
import numpy as np
import torch
import pandas as pd
import os
import yaml
import re

EXPECTED_YAML_STRUCTURE = """
The training.yaml file could not be parsed. Please ensure it follows this structure:

# MLflow Configuration
mlflow:
  tracking_uri: "http://<your_mlflow_tracking_uri>:<port>" # "http://35.184.133.0:5000" for office VM and "http://10.128.0.3:5000" for GCP VM
  experiment_name: "Your_Experiment_Name"
  user: "Your_Username"
  tracking_username: "admin"  # MLflow server username
  tracking_password: "password"  # MLflow server password

# YOLO Configuration (optional if not using YOLO)
yolo_config:
  data_path: "path/to/your/data.yaml"
  cfg_path: "path/to/your/cfg.yaml"
  weights_path: "path/to/your/weights.pt"
"""


def validate_format(input_string):
    valid_models = [
        "YOLOv8",
        "RT-DETR",
        "YOLOv10",
        "YOLO11",
        "TFClassifier",
        "TFSegmentation",
        "PTClassifier",
    ]
    model_pattern = "|".join(valid_models)

    pattern = (
        r"^[A-Za-z0-9]+_"
        r"([A-Za-z0-9]+-?)*_"
        r"[A-Za-z0-9]+(-[A-Za-z0-9]+)*_"
        rf"({model_pattern})$"
    )

    valid_format = bool(re.match(pattern, input_string))

    if not valid_format:
        raise ValueError(
            "Experiment name does not match the required format. CK is not happy! Please ensure it follows this exact format:\n\n"
            "[Project]_[UseCase/Sub-Project]_[Task]_[Model]\n\n"
            "Format Details:\n"
            "   - [Client/Project]: Specifies the client or project, like PSA, ITE, or HDB. This should be an alphanumeric code identifying the client or project.\n"
            "   - [UseCase/Sub-Project]: An alphanumeric field that specifies a particular case or sub-project, such as Case3, Case6, or PMD.\n"
            "   - [Task]: Describes the main object or feature being worked on. For example, use terms like Powder-Caution, Car, SilverValveHandle, or Controller. This field should provide clear context for the work being done.\n"
            f"   - [Model]: Specifies the model or task type, such as {', '.join(valid_models)}. This component indicates the model type and should be alphanumeric without underscores.\n\n"
            "Components must not contain underscores (_), as these are reserved as delimiters. Use hyphens (-) within components if subcategorization is needed.\n\n"
            "Good Example: ITE_Case3_Powder-Caution_YOLOv8\n"
            "Bad Examlples: Case3_Powder-Caution_YOLOv8, ITE_Case3_Powder-Caution_YOLODetector, ITE_Case3_Powder-Caution, ITE_Case3_Powder_Caution_YOLOv8\n\n"
        )

    print("Experiment name format is valid. Good Job! CK loves you! <3")


def load_and_validate_config(file_path="training.yaml"):
    """
    Loads and validates the configuration file.
    Returns a dictionary with the validated configuration values if all checks pass.
    Raises an error if any issues are encountered.
    """
    # Load configuration from training.yaml
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        if not config:
            raise ValueError("The configuration file is empty.")
    except (yaml.YAMLError, FileNotFoundError, ValueError) as e:
        raise ValueError(
            f"Error loading training.yaml: {e}\n\n{EXPECTED_YAML_STRUCTURE}"
        )

    # Validate MLflow configuration
    mlflow_config = config.get("mlflow", {})
    if not mlflow_config:
        raise ValueError(
            f"Missing MLflow configuration in training.yaml.\n\n{EXPECTED_YAML_STRUCTURE}"
        )

    # Required fields for MLflow
    MLFLOW_TRACKING_URI = mlflow_config.get("tracking_uri")
    MLFLOW_EXPERIMENT_NAME = mlflow_config.get("experiment_name")
    USER = mlflow_config.get("user")

    if not (MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME and USER):
        raise ValueError(
            f"Incomplete MLflow configuration in training.yaml.\n\n{EXPECTED_YAML_STRUCTURE}"
        )

    # Check experiment name format
    validate_format(MLFLOW_EXPERIMENT_NAME)

    # Set MLflow credentials as environment variables
    MLFLOW_TRACKING_USERNAME = mlflow_config.get("tracking_username", "admin")
    MLFLOW_TRACKING_PASSWORD = mlflow_config.get("tracking_password", "password")

    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

    # Validate YOLO configuration
    yolo_config = config.get("yolo_config", {})
    if not yolo_config:
        raise ValueError(
            f"Missing YOLO configuration in training.yaml.\n\n{EXPECTED_YAML_STRUCTURE}"
        )

    # Required fields for YOLO
    DATA_PATH = yolo_config.get("data_path") # coco8.yaml
    CFG_PATH = yolo_config.get("cfg_path") # default.yaml
    WEIGHTS_PATH = yolo_config.get("weights_path")
    RESUME = yolo_config.get("resume", False)

    if not (DATA_PATH and CFG_PATH and WEIGHTS_PATH):
        raise ValueError(
            f"Incomplete YOLO configuration in training.yaml.\n\n{EXPECTED_YAML_STRUCTURE}"
        )

    # Return all validated values as a dictionary
    return {
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
        "USER": USER,
        "DATA_PATH": DATA_PATH,
        "CFG_PATH": CFG_PATH,
        "WEIGHTS_PATH": WEIGHTS_PATH,
        "MLFLOW_TRACKING_USERNAME": MLFLOW_TRACKING_USERNAME,
        "MLFLOW_TRACKING_PASSWORq": MLFLOW_TRACKING_PASSWORD,
        "RESUME": RESUME,
    }


# Load and validate configuration
config_values = load_and_validate_config()

# Assign constants from the validated configuration
MLFLOW_TRACKING_URI = config_values["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT_NAME = config_values["MLFLOW_EXPERIMENT_NAME"]
USER = config_values["USER"]
DATA_PATH = config_values["DATA_PATH"]
CFG_PATH = config_values["CFG_PATH"]
WEIGHTS_PATH = config_values["WEIGHTS_PATH"]
RESUME = config_values["RESUME"]


# Set the MLflow tracking URI to the local server
print(f"Trying to connect to MLflow at tracking URI: {MLFLOW_TRACKING_URI}")

try:
    # Attempt to set the MLflow tracking URI and experiment name
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    use_mlflow = True
except Exception as e:
    # Print the error message if an exception occurs
    print(f"Error setting MLflow tracking: {str(e)}")
    print("Continuing without MLflow tracking...")
    use_mlflow = False

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME
mlflow.autolog(disable=True)
absolute_data_path = os.path.abspath(DATA_PATH)
absolute_cfg_path = os.path.abspath(CFG_PATH)


# Callbacks
def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def log_model(model, run_id, model_name):
    """
    Log the model to MLflow.
    """
    try:
        mlflow.pytorch.log_model(str(model.trainer.best), model_name)
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
    except Exception as e:
        print(f"An error occurred while logging model: {e}")


def safe_log_param(key, value):
    """Safely log a parameter to MLflow."""
    try:
        mlflow.log_param(key, value)
    except Exception as e:
        print(f"An error occurred while logging parameter '{key}': {e}")


def safe_log_artifact(file_path, artifact_path=None):
    """Safely log an artifact to MLflow."""
    try:
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
    except Exception as e:
        print(f"An error occurred while logging artifact '{file_path}': {e}")


def safe_log_metric(key, value, step=None):
    """Safely log a metric to MLflow."""
    try:
        mlflow.log_metric(key, value, step=step)
    except Exception as e:
        print(f"An error occurred while logging metric '{key}': {e}")


def safe_log_metrics(metrics, step=None):
    """Safely log multiple metrics to MLflow."""
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        print(f"An error occurred while logging metrics: {e}")


def safe_log_dataset(dataset, context):
    """Safely log a dataset to MLflow."""
    try:
        mlflow.log_input(dataset, context)
    except Exception as e:
        print(f"An error occurred while logging dataset: {e}")


def safe_log_dict(dict, path):
    """Safely log a dataset to MLflow."""
    try:
        mlflow.log_dict(dict, path)
    except Exception as e:
        print(f"An error occurred while logging dataset: {e}")


def pre_training(model):
    """
    Log the initial parameters at the start of training, such as:
    - Weights path
    - Configuration artifacts
    """
    # Log the model used (check transfer learning)

    mlflow.set_tag("user", USER)

    weights_path = model.ckpt_path
    safe_log_param("weights_path", weights_path)

    # Log the configuration artifacts
    safe_log_artifact(absolute_data_path, artifact_path="config")
    safe_log_artifact(absolute_cfg_path, artifact_path="config")


def on_train_start(trainer):
    """
    Log parameters at the start of training:
    - Length of the training dataset
    - Length of the validation dataset
    - Number of epochs
    """

    train_image_paths = trainer.train_loader.dataset.im_files
    test_image_paths = trainer.test_loader.dataset.im_files

    test_label_paths = trainer.test_loader.dataset.label_files
    train_label_paths = trainer.train_loader.dataset.label_files

    # dataset = pd.DataFrame(
    #     {
    #         "image": train_image_paths + test_image_paths,
    #         "label": train_label_paths + test_label_paths,
    #         "type": ["train"] * len(train_image_paths)
    #         + ["test"] * len(test_image_paths),
    #     }
    # )
    # safe_log_dict(
    #     {
    #         "train": train_image_paths,
    #         "train_labels": train_label_paths,
    #         "val": test_image_paths,
    #         "val_labels": test_label_paths,
    #     },
    #     "dataset.json",
    # )

    # dataset = mlflow.data.from_pandas(
    #     dataset, source=LocalArtifactDatasetSource(str(trainer.data["path"]))
    # )
    # safe_log_dataset(dataset, "train")

    mlflow.set_tag("mlflow.runName", str(trainer.save_dir).split("/")[-1])
    # Log the length of the training dataset
    safe_log_param("train_dataset_length", len(train_image_paths))

    # Log the length of the validation dataset
    safe_log_param("val_dataset_length", len(test_image_paths))

    # Log the number of epochs
    safe_log_param("epochs", trainer.epochs)

    save_dir = model.trainer.save_dir
    artifacts = os.listdir(save_dir)
    artifacts = [
        artifact
        for artifact in artifacts
        if not os.path.isdir(os.path.join(save_dir, artifact))
    ]

    # Log the artifacts produced after training
    for artifact in artifacts:
        safe_log_artifact(
            os.path.join(save_dir, artifact),
        )


def on_train_epoch_end(trainer):
    """
    Log parameters at the end of each epoch:
    - Learning rate
    - Loss
    - Current weights
    - Results
    """

    # Removed cuz this is covered by YOLO's Auto Logging
    metrics = {
        **sanitize_dict(trainer.lr),
        **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
    }
    safe_log_metrics(metrics, step=trainer.epoch)


def on_fit_epoch_end(trainer):
    # Removed cuz this is covered by YOLO's Auto Logging
    try:
        for k, v in trainer.metrics.items():
            safe_log_metric(
                k.split("/")[-1].replace("(B)", "_training"),
                float(v),
                step=trainer.epoch,
            )
    except Exception as e:
        print(f"An error occurred while accessing trainer metrics: {e}")

    # Log the current weights
    try:
        for file in os.listdir(str(trainer.best.parent)):
            safe_log_artifact(
                os.path.join(str(trainer.best.parent), file),
                artifact_path="weights",
            )
    except Exception as e:
        print(f"An error occurred while accessing weights directory: {e}")

    # Log the result.csv if it exists
    if os.path.exists(str(trainer.csv)):
        safe_log_artifact(str(trainer.csv))


def post_training(model):
    """
    Log parameters at the end of training:
    - Best weights
    - Artifacts produced after training
    """
    # Log the best weights
    try:
        for file in os.listdir(str(model.trainer.best.parent)):
            safe_log_artifact(
                os.path.join(str(model.trainer.best.parent), file),
                artifact_path="final_weights",
            )
    except Exception as e:
        print(f"An error occurred while accessing best weights directory: {e}")

    # List and filter out directories from artifacts
    try:
        save_dir = model.trainer.save_dir
        artifacts = os.listdir(save_dir)
        artifacts = [
            artifact
            for artifact in artifacts
            if not os.path.isdir(os.path.join(save_dir, artifact))
        ]

        # Log the artifacts produced after training
        for artifact in artifacts:
            safe_log_artifact(
                os.path.join(save_dir, artifact),
            )
    except Exception as e:
        print(f"An error occurred while accessing or logging artifacts: {e}")


# def log_onnx(model, name):
#     """
#     Convert the model to ONNX format.
#     """
#     try:
#         res = model.export(format="onnx")
#         safe_log_artifact(res, artifact_path="onnx_model")

#         onnx_model = onnx.load(res)

#         inference_model = ort.InferenceSession(res)
#         input_name = inference_model.get_inputs()[0].name
#         input_shape = inference_model.get_inputs()[0].shape

#         model_input = {input_name: torch.randn(*input_shape).numpy()}
#         model_output = inference_model.run(None, model_input)

#         model_sign = infer_signature(
#             model_input=model_input, model_output=model_output[0]
#         )

#         mlflow.onnx.log_model(
#             onnx_model,
#             artifact_path="onnx_model",
#             registered_model_name=name,
#             signature=model_sign,
#         )

#     except Exception as e:
#         print(f"An error occurred while converting the model to ONNX: {e}")

def on_pretrain_routine_end(trainer):
    pass 

if use_mlflow:
    if RESUME:
        # get the last run id from mlflow
        try:
            train_name = WEIGHTS_PATH.split("/")[-3] # train4 or smt
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

            experiment_id = experiment.experiment_id
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.mlflow.runName = '{train_name}'"
            )

            if len(runs) == 0:
                raise ValueError("No runs found with the specified name")

            last_run_id = runs.sort_values("start_time", ascending=False).iloc[0]["run_id"]
        
        except Exception as e:
            print(f"An error occurred while resuming the run: {e}")
        
        with mlflow.start_run(run_id=last_run_id) as run:
            model = YOLO(WEIGHTS_PATH)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_start", on_pretrain_routine_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            # model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

            results = model.train(resume=True)

            # post_training(model)

            # log_onnx(model, MLFLOW_EXPERIMENT_NAME)
    else:
        with mlflow.start_run() as run:

            # Log the dummy model
            class DummyModel(mlflow.pyfunc.PythonModel):
                def predict(self, context, model_input):
                    return [1] * len(model_input)

            mlflow.pyfunc.log_model(
                artifact_path="dummy_model",
                python_model=DummyModel(),
                registered_model_name=MLFLOW_EXPERIMENT_NAME,
            )
            print(f"Dummy model successfully logged")

            model = YOLO(WEIGHTS_PATH)

            pre_training(model)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

            results = model.train(data=absolute_data_path, cfg=absolute_cfg_path)

            # post_training(model)

            # log_onnx(model, MLFLOW_EXPERIMENT_NAME)
else:
    print("\n\n\nATTENTION: MLFLOW NOT BEING USED\n\n\n")
    model = YOLO(WEIGHTS_PATH)
    results = model.train(data=absolute_data_path, cfg=absolute_cfg_path)
