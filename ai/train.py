"""Train the OK / NOT OK classifier on UV magnetic particle inspection images.

Expects:  data/OK/*.jpg  and  data/NON_OK/*.jpg
Usage:    python train.py --model vgg16 --epochs 90
"""
import argparse

from tensorflow.keras.applications import EfficientNetB0, VGG16, efficientnet, vgg16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
CLASSES = ["OK", "NON_OK"]  # index 1 = NON_OK, so the sigmoid output is P(defect)

# backbone, matching preprocessing, number of trainable top layers
BACKBONES = {
    "vgg16": (VGG16, vgg16.preprocess_input, 10),
    "efficientnetb0": (EfficientNetB0, efficientnet.preprocess_input, 20),
}


def build_model(name):
    backbone, _, trainable = BACKBONES[name]
    base = backbone(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    for layer in base.layers[:-trainable]:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    model = Model(base.input, Dense(1, activation="sigmoid")(x))
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_split(name, subset, **augmentation):
    generator = ImageDataGenerator(
        preprocessing_function=BACKBONES[name][1], validation_split=0.2, **augmentation)
    return generator.flow_from_directory(
        "data", target_size=IMG_SIZE, batch_size=BATCH_SIZE, classes=CLASSES,
        class_mode="binary", subset=subset, shuffle=subset == "training", seed=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=BACKBONES, default="vgg16")
    parser.add_argument("--epochs", type=int, default=90)
    args = parser.parse_args()

    # augmentation on the training images only
    train = load_split(args.model, "training", rotation_range=30, zoom_range=0.2,
                       width_shift_range=0.2, height_shift_range=0.2,
                       brightness_range=[0.6, 1.4], horizontal_flip=True)
    val = load_split(args.model, "validation")
    assert train.class_indices == {"OK": 0, "NON_OK": 1}

    model = build_model(args.model)
    model.fit(train, validation_data=val, epochs=args.epochs)
    loss, accuracy = model.evaluate(val, verbose=0)
    print(f"validation accuracy: {accuracy:.1%} on {val.samples} images")
    model.save(f"{args.model}.keras")


if __name__ == "__main__":
    main()
