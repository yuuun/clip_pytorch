import os
from tensorflow.keras.utils import get_file
class Dataset(object):
    def __init__(self):
        # hide
        root_dir = "datasets"
        annotations_dir = os.path.join(root_dir, "annotations")
        images_dir = os.path.join(root_dir, "train2014")
        annotation_file = os.path.join(annotations_dir, "captions_train2014.json")

        # Download caption annotation files
        if not os.path.exists(annotations_dir):
            annotation_zip = get_file(
                "captions.zip",
                cache_dir=os.path.abspath("."),
                origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                extract=True,
            )
            os.remove(annotation_zip)

        # Download image files
        if not os.path.exists(images_dir):
            image_zip = get_file(
                "train2014.zip",
                cache_dir=os.path.abspath("."),
                origin="http://images.cocodataset.org/zips/train2014.zip",
                extract=True,
            )
            os.remove(image_zip)

        print("Dataset is downloaded and extracted successfully.")