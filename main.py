from tensorflow import keras
from utils import generate_recursive_indices
from model import Block, Cell, Operation, OperationType
import numpy as np
import tensorflow as tf
import datetime


hparams = {
    'num_conv_filters': 12,
    'num_blocks': 5,
    'num_cells': 12,
    'num_reduction_cells': 2,
    'dropout_prob': 0.5,
    'filter_scaling_rate': 1.0,
    # stem_multiplier=1.0,
    # drop_path_keep_prob=1.0,
    # num_conv_filters=44,
    # use_aux_head=1,
    # data_format='NHWC',
    # skip_reduction_layer_input=0,
    # total_training_steps=250000,
    # use_bounded_activation=False,
}

rng = np.random.default_rng()


def generate_operation(activation='relu') -> Operation:
    op_type = rng.choice(list(OperationType), 1)[0]
    num_conv_filters = hparams['num_conv_filters']
    filter_scaling_rate = hparams['filter_scaling_rate']
    return Operation(op_type, init_filters=num_conv_filters,
                     filter_scaling_rate=filter_scaling_rate, activation=activation)


def generate_block() -> Block:
    op0 = generate_operation()
    op1 = generate_operation()
    return Block(op0, op1)


def generate_cell(reduce=False) -> Cell:
    num_blocks = hparams['num_blocks']
    blocks = [generate_block()
              for _ in range(num_blocks)]

    block_inputs = generate_recursive_indices(2, 2, num_blocks)
    return Cell(blocks, block_inputs, reduce=reduce)


def generate_model(input_shape: tuple[int, int, int],
                   num_classes: int, hparams: dict[str, int], preprocessing: list[keras.layers.Layer] = [], rescaling=True) -> keras.Model:
    num_cells = hparams['num_cells']
    num_reduction_cells = hparams['num_reduction_cells']
    dropout_prob = hparams['dropout_prob']
    num_between = int((num_cells - num_reduction_cells) /
                      (num_reduction_cells + 1))

    normal_cell = generate_cell(reduce=False)
    reduce_cell = generate_cell(reduce=True)
    cells: list[Cell] = []
    while num_reduction_cells > 0:
        cells += [normal_cell for _ in range(num_between)]
        cells.append(reduce_cell)
        num_reduction_cells -= 1

    num_left = num_cells - len(cells)
    if num_left > 0:
        cells += [normal_cell for _ in range(num_left)]

    input_indices = generate_recursive_indices(1, 2, num_cells)
    inputs = keras.Input(input_shape)
    x = keras.Sequential(preprocessing)(inputs)
    hiddens = [x]

    for i, cell in enumerate(cells):
        input_tup = input_indices[i]
        input0 = hiddens[input_tup[0]]
        input1 = hiddens[input_tup[1]]

        cell_name = ("cell_%d_reduce" if cell.reduce else "cell_%d_normal") % i
        hidden = cell(input0, input1, name=cell_name)
        print(hidden.shape)
        hiddens.append(hidden)

    # Connect specific layers to the final cell to get the result
    final_hidden = hiddens[-1]
    x = keras.layers.GlobalAvgPool2D()(final_hidden)
    x = keras.layers.Dropout(dropout_prob)(x)
    outputs = keras.layers.Dense(num_classes)(x)

    model = keras.Model(inputs, outputs, name="PeanutButterNet")
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    image_shape = tuple(x_train.shape[1:])
    num_classes = 10
    batch_size = 32

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train))
    train_ds = train_ds.batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(32)

    preprocessors = [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ]

    model = generate_model(
        image_shape,
        num_classes,
        hparams,
        preprocessing=preprocessors)
    model.summary()

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + now_str
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "cifar10_classification_%s.h5" % now_str, save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15)

    callbacks = [tensorboard_cb, checkpoint_cb, early_stopping_cb]
    epochs = 50
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds)

    predictions = model.predict(x_test)
    comparision = tf.math.equal(predictions, y_test)
    comparision = tf.squeeze(comparision)
    test_acc = tf.reduce_sum(tf.cast(comparision, dtype=tf.float32)).numpy()
    test_acc = test_acc / len(y_test)
    print("Config:")
    print(hparams)
    print("Log dir: " + log_dir)
    print("Test accuracy is %.5f percent." % test_acc)
