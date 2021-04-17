from tensorflow import keras
from utils import generate_recursive_indices
from model import Block, Cell, Operation, OperationType
import numpy as np


hparams = {
    'num_conv_filters': 32,
    'num_blocks': 5,
    'num_cells': 10,
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
                   num_classes: int, hparams: dict[str, int]) -> keras.Model:
    num_cells = hparams['num_cells']
    num_reduction_cells = hparams['num_reduction_cells']
    dropout_prob = hparams['dropout_prob']
    num_between = int((num_cells - num_reduction_cells) /
                      (num_reduction_cells + 1))

    cells: list[Cell] = []
    while num_reduction_cells > 0:
        cells += [generate_cell(reduce=False) for _ in range(num_between)]
        cells.append(generate_cell(reduce=True))
        num_reduction_cells -= 1

    num_left = num_cells - len(cells)
    if num_left > 0:
        cells += [generate_cell(reduce=False) for _ in range(num_left)]

    input_indices = generate_recursive_indices(1, 2, num_cells)
    inputs = keras.Input(input_shape)
    hiddens = [inputs]

    for i, cell in enumerate(cells):
        input_tup = input_indices[i]
        input0 = hiddens[input_tup[0]]
        input1 = hiddens[input_tup[1]]
        # if input0.shape[1:3] != input1.shape[1:3]:
        #     target_shape = max(input0.shape[1:3], input1.shape[1:3])
        #     if input0.shape != target_shape:
        #         input0 = keras.layers.experimental.preprocessing.Resizing(target_shape[0], target_shape[1])(input0)
        #     else:
        #         input1 = keras.layers.experimental.preprocessing.Resizing(target_shape[0], target_shape[1])(input1)

        hidden = cell(input0, input1)
        hiddens.append(hidden)

    # Connect specific layers to the final cell to get the result
    final_hidden = hiddens[-1]
    x = keras.layers.GlobalAvgPool2D()(final_hidden)
    x = keras.layers.Dropout(dropout_prob)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')

    model = keras.Model(inputs, outputs, name="PeanutButterNet")
    return model


model = generate_model((256, 256, 1), 10, hparams)
model.summary()
