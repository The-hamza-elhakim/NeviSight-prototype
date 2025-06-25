from tensorflow.keras.models import load_model
from keras_unet_collection import losses

model = load_model("models/best_model_128.keras", custom_objects={
    'iou_seg': losses.iou_seg,
    'dice': losses.dice  # Add this too if your model uses it
})

print(model.input_shape)
