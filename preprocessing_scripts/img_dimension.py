import cv2

# Load the image
image_path = "box_datasets/box_dataset_final/train/images/bdb8fc4bd57bc0cb_jpg.rf.339e8dcdc64bc6753cdb53a2a1fa128c.jpg"
image = cv2.imread(image_path)

if image is not None:
    # Get the dimensions of the image
    height, width, channels = image.shape
    print(f"Width: {width}, Height: {height}, Channels: {channels}")
else:
    print(f"Could not load image at {image_path}")