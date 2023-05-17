import os
import h5py
from keras.preprocessing.image import ImageDataGenerator


def get_data_generator(batch_size=32) -> tuple:
    train_dir = "../data/Flower Classification/Training Data"
    test_dir = "../data/Flower Classification/Testing Data"
    validation_dir = "../data/Flower Classification/Validation Data"

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        rescale=1.0 / 255,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=42,
    )
    print("Data generators created successfully!")
    return train_generator, validation_generator, test_generator


def save_results(args, results):
    os.makedirs(f"../results/{args.model}", exist_ok=True)

    filename = f"../results/{args.model}/{args.model}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.epochs}.h5"

    with h5py.File(filename, "w") as f:
        for key, value in results.items():
            f.create_dataset(key, data=value)

    print("Results saved successfully!")


def get_num_of_models(model_name):
    models = os.listdir(f"../models/{model_name}")
    return len(models)
