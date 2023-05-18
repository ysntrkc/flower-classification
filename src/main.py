import os
import time
import tensorflow as tf

from model.CNN import CNN
from utils.options import args_parser
from utils.kaggle import download_dataset
from utils.utils import get_data_generator, save_results, get_num_of_models

from keras.applications import ResNet50


def train(
    args,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    lr_scheduler,
    optimizer,
    loss_fn,
):
    os.makedirs(f"../models/{args.model}", exist_ok=True)
    model_num = get_num_of_models(args.model)
    max_acc = 0
    model.save_weights(f"../models/{args.model}/{args.model}_{model_num}.h5")
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    test_loss, test_acc = [], []

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss_epoch = tf.keras.metrics.Mean()
        train_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for x_batch, y_batch in train_dataloader:
            with tf.GradientTape() as tape:
                pred = model(x_batch, training=True)
                loss_value = loss_fn(y_batch, pred)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss_epoch.update_state(loss_value)
            train_acc_epoch.update_state(y_batch, pred)

        train_loss.append(train_loss_epoch.result())
        train_acc.append(train_acc_epoch.result())

        val_loss_epoch = tf.keras.metrics.Mean()
        val_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()

        # Validation loop
        for x_batch, y_batch in val_dataloader:
            pred = model(x_batch, training=False)
            loss_value = loss_fn(y_batch, pred)

            val_loss_epoch.update_state(loss_value)
            val_acc_epoch.update_state(y_batch, pred)

        val_loss.append(val_loss_epoch.result())
        val_acc.append(val_acc_epoch.result())

        lr_scheduler.step(epoch)

        if val_acc[-1] > max_acc:
            max_acc = val_acc[-1]
            model.save_weights(f"../models/{args.model}/{args.model}_{model_num}.h5")

        # Test loop
        test_loss_epoch = tf.keras.metrics.Mean()
        test_acc_epoch = tf.keras.metrics.SparseCategoricalAccuracy()

        for x_batch, y_batch in test_dataloader:
            pred = model(x_batch, training=False)
            loss_value = loss_fn(y_batch, pred)

            test_loss_epoch.update_state(loss_value)
            test_acc_epoch.update_state(y_batch, pred)

        test_loss.append(test_loss_epoch.result())
        test_acc.append(test_acc_epoch.result())

        end_time = time.time()
        print(
            f"Epoch {epoch+1:>2}/{args.epochs} | Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc[-1]:.4f}",
            end=" | ",
        )
        print(
            f"Val Loss: {val_loss[-1]:.4f} | Val Acc: {val_acc[-1]:.4f}",
            end=" | ",
        )
        print(
            f"Test Loss: {test_loss[-1]:.4f} | Test Acc: {test_acc[-1]:.4f}",
            end=" | ",
        )
        print(f"Time taken: {end_time - start_time:.2f} seconds")

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main():
    args = args_parser()
    download_dataset()

    args.device = tf.device(
        f"/GPU:{args.device_id}" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    )

    train_generator, validation_generator, test_generator = get_data_generator(
        args.batch_size
    )

    if (args.model).lower() == "cnn":
        model = CNN(args.num_classes)
    elif (args.model).lower() == "resnet":
        model = ResNet50(
            include_top=False,
            weights="imagenet",
            classes=args.num_classes,
        )
    else:
        raise ValueError(f"Model {args.model} not found!")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    exp_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    history = train(
        args,
        model,
        train_generator,
        validation_generator,
        test_generator,
        exp_lr_scheduler,
        optimizer,
        loss_fn,
    )

    save_results(args, history)


if __name__ == "__main__":
    main()
