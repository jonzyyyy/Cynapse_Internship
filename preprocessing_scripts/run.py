import os
import yaml
import shutil

"""
IMPORTANT!!!!!!!!!!!!!!!!!!!!

PLEASE CHANGE THE FILE LOCATIONS AT THE BOTTOM OF THE FILE + 
Modify the original classes, selected classes, destination path and dataset limits of the YAML
file correspondingly
"""

class DatasetProcessor:
    """
    Processes datasets for YOLO training based on a YAML configuration.

    This class filters images and label files for selected classes,
    applies dataset limits and updates class IDs in label files.
    """

    def __init__(self, config_path: str, source_paths: dict) -> None:
        """
        Initialise the dataset processor.

        Parameters:
            config_path (str): Path to the YAML configuration file.
            source_paths (dict): Dictionary with source paths for 'train', 'val', and 'test'.
        """
        self.config_path = config_path
        self.source_paths = source_paths
        self.config = self._load_config()
        self.class_names = self.config['class_names']
        self.initial_selected_classes = self.config['selected_classes']
        self.destination_paths = self.config['destination_paths']
        self.dataset_limits = self.config['dataset_limits']

        self._create_destination_directories()

    def _load_config(self) -> dict:
        """
        Load the YAML configuration file.

        Returns:
            dict: The configuration dictionary.
        """
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def _create_destination_directories(self) -> None:
        """
        Create the destination directories for images and labels.
        """
        for dataset in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.destination_paths[dataset], 'images'), exist_ok=True)
            if dataset != 'test':
                os.makedirs(os.path.join(self.destination_paths[dataset], 'labels'), exist_ok=True)

    def _get_limit(self, class_name: str, limits: dict) -> str:
        """
        Return the limit for a given class from the provided limits dictionary.
        If a specific limit is not found, try the 'all' key; if neither exists, default to 'all'.

        Parameters:
            class_name (str): The class name.
            limits (dict): The limits dictionary for the dataset.

        Returns:
            str: The limit as a string, either a number (in string form) or 'all'.
        """
        return limits.get(class_name, limits.get('all', 'all'))

    def _process_train_val(self, dataset: str, selected_classes: dict) -> None:
        """
        Process training or validation data by filtering annotations,
        copying files and updating class IDs.

        Parameters:
            dataset (str): Either 'train' or 'val'.
            selected_classes (dict): Mapping of new class IDs (int) to class names.
        """
        images_path = self.source_paths[dataset]['images']
        labels_path = self.source_paths[dataset].get('labels')
        dest_images_path = os.path.join(self.destination_paths[dataset], 'images')
        dest_labels_path = os.path.join(self.destination_paths[dataset], 'labels')

        label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

        # Initialise counters: keys are class names.
        class_counters = {class_name: 0 for _, class_name in selected_classes.items()}
        # Make a copy of selected_classes to track classes that haven't hit their quota.
        classes_not_hit_quota_yet = selected_classes.copy()
        limits = self.dataset_limits[dataset]

        for label_file in label_files:
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            src_image_path = os.path.join(images_path, image_file)
            src_label_path = os.path.join(labels_path, label_file)

            if not os.path.exists(src_image_path):
                continue

            # Read and filter annotations.
            with open(src_label_path, 'r') as file:
                lines = file.readlines()

            filtered_lines = []
            contains_needed_class = False
            for line in lines:
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                class_name = self.class_names.get(class_id)
                if class_name is None:
                    print(f"Warning: Class ID {class_id} not found in class_names")
                    continue
                # Check if this annotation belongs to one of the selected classes.
                if class_name in selected_classes.values() or class_name in classes_not_hit_quota_yet.values():
                    filtered_lines.append(line)
                    contains_needed_class = True

            if not contains_needed_class:
                print(f"Skipping: No selected classes found in {label_file}")
                continue

            # Copy the corresponding image.
            dest_image_path = os.path.join(dest_images_path, image_file)
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied image: {image_file} from {src_image_path} to {dest_image_path}")
            print(f"Destination file exists: {os.path.exists(dest_image_path)}")

            # Update the label file with new class IDs.
            dest_label_path = os.path.join(dest_labels_path, label_file)
            with open(dest_label_path, 'w') as file:
                for line in filtered_lines:
                    parts = line.split()
                    try:
                        old_class_id = int(parts[0])
                    except ValueError:
                        continue
                    if old_class_id in self.class_names:
                        old_class_name = self.class_names[old_class_id]
                        for new_class_id, new_class_name in selected_classes.items():
                            if old_class_name == new_class_name:
                                file.write(f"{new_class_id} {' '.join(parts[1:])}\n")
                                break

            # Update counters and check against limits.
            updated = False
            for line in filtered_lines:
                parts = line.split()
                try:
                    old_class_id = int(parts[0])
                except ValueError:
                    continue
                old_class_name = self.class_names.get(old_class_id)
                if old_class_name in class_counters and not updated:
                    limit = self._get_limit(old_class_name, limits)
                    if limit == 'all' or class_counters[old_class_name] < int(limit):
                        class_counters[old_class_name] += 1
                        updated = True

            # Remove classes that have reached their limit.
            filtered_classes = {}
            for new_id, class_name in classes_not_hit_quota_yet.items():
                limit = self._get_limit(class_name, limits)
                if limit == 'all' or class_counters[class_name] < int(limit):
                    filtered_classes[new_id] = class_name
            classes_not_hit_quota_yet = filtered_classes

            # Update selected_classes to only include classes still under limit.
            selected_classes = {new_id: class_name for new_id, class_name in selected_classes.items()
                                if new_id in classes_not_hit_quota_yet}

            # Break if no more classes are left to process.
            if not selected_classes:
                break

    def _parse_test_limit(self) -> int:
        """
        Parse the test limit from dataset_limits. If no valid limit is found, return infinity.

        Returns:
            int: The maximum number of test images to process.
        """
        test_limits = self.dataset_limits.get('test', {})
        value = test_limits.get('none', '')
        try:
            total_limit = int(value)
        except (ValueError, TypeError):
            total_limit = float('inf')
        return total_limit

    def _process_test(self, dataset: str) -> None:
        """
        Process the test dataset by copying images up to the total limit.

        Parameters:
            dataset (str): 'test'
        """
        images_path = self.source_paths[dataset]['images']
        dest_images_path = os.path.join(self.destination_paths[dataset], 'images')
        file_names = os.listdir(images_path)
        total_limit = self._parse_test_limit()
        for count, filename in enumerate(file_names):
            if count >= total_limit:
                break
            src_image_path = os.path.join(images_path, filename)
            dest_image_path = os.path.join(dest_images_path, filename)
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied image: {filename}")

    def process_dataset(self, dataset: str, selected_classes: dict) -> None:
        """
        Process a given dataset.

        Parameters:
            dataset (str): One of 'train', 'val' or 'test'.
            selected_classes (dict): Mapping of new class IDs to class names.
        """
        if dataset in ['train', 'val']:
            self._process_train_val(dataset, selected_classes)
        elif dataset == 'test':
            self._process_test(dataset)
        else:
            raise ValueError(f"Invalid dataset type: {dataset}")

    def run(self) -> None:
        """
        Run the processing for all datasets: train, validation and test.
        """
        for dataset_name in ['train', 'val', 'test']:
            print(f"\n\nProcessing dataset: {dataset_name}")
            # Use a fresh copy of the selected classes for each dataset.
            current_selected_classes = self.initial_selected_classes.copy()
            self.process_dataset(dataset_name, current_selected_classes)
            print(f"Finished processing dataset: {dataset_name}")


def main() -> None:
    # Define the source paths.
    source_paths = {
        'train': {
            'images': 'scooter_training_dataset/train/images/',   # Replace with path to train images
            'labels': 'scooter_training_dataset/train/labels/'      # Replace with path to train labels
        },
        'val': {
            'images': 'scooter_training_dataset/val/images/',     # Replace with path to val images
            'labels': 'scooter_training_dataset/val/labels/'        # Replace with path to val labels
        },
        'test': {
            'images': 'scooter_training_dataset/test/'      # Replace with path to test images
        }
    }

    processor = DatasetProcessor(config_path='yaml/config.yaml', source_paths=source_paths)
    processor.run()


if __name__ == '__main__':
    main()