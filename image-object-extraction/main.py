# https://github.com/ayoolaolafenwa/PixelLib/blob/master/Tutorials/Pytorch_image_instance_segmentation.md

import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("../pixellib_models/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(car=True)
results, output = ins.segmentImage(
    "image.jpg",
    segment_target_classes=target_classes,
    show_bboxes=True,
    extract_segmented_objects=True,
    save_extracted_objects=True,  # For some reason all recognized objects are being saved, not just the target_classes
    extract_from_box=True,
    output_image_name="output_image.jpg",
)
print(results["object_counts"])
