import time
from pixellib.torchbackend.instance import instanceSegmentation

# resnet50
ins = instanceSegmentation()

# Normal
ins.load_model("../pixellib_models/pointrend_resnet50.pkl")
start = time.time()
ins.segmentImage("image.jpg", show_bboxes=True, output_image_name="output_image_50.jpg")
print("normal: ", time.time() - start)

# Fast
ins.load_model("../pixellib_models/pointrend_resnet50.pkl", detection_speed="fast")
start = time.time()
ins.segmentImage(
    "image.jpg", show_bboxes=True, output_image_name="output_image_50_fast.jpg"
)
print("fast: ", time.time() - start)

# Rapid
ins.load_model("../pixellib_models/pointrend_resnet50.pkl", detection_speed="rapid")
start = time.time()
ins.segmentImage(
    "image.jpg", show_bboxes=True, output_image_name="output_image_50_rapid.jpg"
)
print("rapid: ", time.time() - start)

# resnet101
ins = instanceSegmentation()

# Normal
ins.load_model(
    "../pixellib_models/pointrend_resnet101.pkl", network_backbone="resnet101"
)
start = time.time()
ins.segmentImage(
    "image.jpg", show_bboxes=True, output_image_name="output_image_101.jpg"
)
print("normal: ", time.time() - start)

# Fast
ins.load_model(
    "../pixellib_models/pointrend_resnet101.pkl",
    network_backbone="resnet101",
    detection_speed="fast",
)
start = time.time()
ins.segmentImage(
    "image.jpg", show_bboxes=True, output_image_name="output_image_101_fast.jpg"
)
print("fast: ", time.time() - start)

# Rapid
ins.load_model(
    "../pixellib_models/pointrend_resnet101.pkl",
    network_backbone="resnet101",
    detection_speed="rapid",
)
start = time.time()
ins.segmentImage(
    "image.jpg", show_bboxes=True, output_image_name="output_image_101_rapid.jpg"
)
print("rapid: ", time.time() - start)
