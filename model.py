from utils import extract_kernel_sizes
from enum import Enum
import tensorflow as tf
import tensorflow.keras as keras


class CellType(Enum):
    NORMAL = 1
    REDUCTION = 2


class OperationType(Enum):
    IDENTITY = 1
    CONV_1X3_3X1 = 2
    CONV_1X7_7X1 = 3
    CONV_1X1 = 4
    CONV_3X3 = 5
    AVG_POOLING_3X3 = 6
    MAX_POOLING_3X3 = 7
    MAX_POOLING_5X5 = 8
    MAX_POOLING_7x7 = 9
    SEP_CONV_3X3 = 10
    SEP_CONV_5X5 = 11
    SEP_CONV_7X7 = 12
    DILATED_CONV_3X3 = 13


class Operation():
    def __init__(self, op_type: OperationType,
                 init_filters: int = 32, filter_scaling_rate: int = 2, activation: str = 'relu') -> None:
        self.op_type = op_type
        self.init_filters = int(init_filters)
        self.filter_scaling_rate = filter_scaling_rate
        self.activation = activation
        self.ops: list[tf.keras.layers.Layer] = []

    def __call__(self, hidden: tf.Tensor, reduce=False, **kwargs) -> tf.Tensor:
        op_name = self.op_type.name.lower()
        if self.op_type == OperationType.IDENTITY:
            pass
        else:
            stride = 2 if reduce else 1
            kernel_sizes = extract_kernel_sizes(self.op_type.name)
            dilation_rate = 2 if "dilated" in op_name and stride == 1 else 1
            filters = self.init_filters \
                if self.init_filters % hidden.shape[-1] == 0 \
                else int(hidden.shape[-1] * self.filter_scaling_rate)
            if "conv" in op_name:
                if "sep" in op_name:
                    self.ops = [
                        keras.layers.SeparableConv2D(
                            filters,
                            kernel,
                            strides=stride,
                            padding='same',
                            dilation_rate=dilation_rate,
                            activation=self.activation, **kwargs) for kernel in kernel_sizes]
                else:
                    self.ops = [
                        keras.layers.Conv2D(
                            filters,
                            kernel,
                            strides=stride,
                            padding='same',
                            dilation_rate=dilation_rate,
                            activation=self.activation, **kwargs) for kernel in kernel_sizes]
            elif "max_pool" in op_name:
                self.ops = [
                    keras.layers.MaxPool2D(
                        pool_size=pool_size,
                        padding='same',
                        strides=stride) for pool_size in kernel_sizes]
            elif "avg_pool" in op_name:
                self.ops = [
                    keras.layers.AvgPool2D(
                        pool_size=pool_size,
                        padding='same',
                        strides=stride) for pool_size in kernel_sizes]
            else:
                raise ValueError("Operation type not supported")

        for op in self.ops:
            hidden = op(hidden)

        return hidden


class Block():
    def __init__(self, op0: Operation, op1: Operation) -> None:
        self.op0 = op0
        self.op1 = op1

    def __call__(self, hidden0: tf.Tensor, hidden1: tf.Tensor, reduce=False,
                 **kwargs) -> tf.Tensor:
        if hidden0.shape[1:3] == hidden1.shape[1:3]:
            hidden0 = self.op0(hidden0, reduce=reduce, **kwargs)
            hidden1 = self.op1(hidden1, reduce=reduce, **kwargs)
        else:
            reduce0 = hidden0.shape[1] > hidden1.shape[1]
            hidden0 = self.op0(hidden0, reduce=reduce0, **kwargs)
            hidden1 = self.op1(hidden1, reduce=not reduce0, **kwargs)

        if hidden0.shape[-1] == hidden1.shape[-1]:
            return keras.layers.add([hidden0, hidden1])
        else:
            return keras.layers.concatenate([hidden0, hidden1], axis=-1)


class Cell():
    def __init__(self, blocks: list[Block],
                 connections: list[tuple[int, int]],
                 reduce=False) -> None:
        assert len(blocks) == len(connections)
        self.blocks = blocks
        self.connections = connections
        self.reduce = reduce

    def __call__(self, hidden0: tf.Tensor, hidden1: tf.Tensor,
                 **kwargs) -> tf.Tensor:
        counter = 1
        connections = self.connections.copy()
        hiddens = [hidden0, hidden1]
        outputs: list[tf.Tensor] = []
        while connections:
            candidate_idx = [all([x <= counter for x in c])
                             for c in connections].index(True)
            block = self.blocks[candidate_idx]
            block_input = connections[candidate_idx]
            reduce = self.reduce and block_input in [
                (0, 0), (0, 1), (1, 0), (1, 1)]
            entry = block(hiddens[block_input[0]],
                          hiddens[block_input[1]],
                          reduce=reduce, **kwargs)

            connections.pop(candidate_idx)
            hiddens.append(entry)
            outputs.append(entry)
            counter += 1

        return keras.layers.concatenate(outputs, axis=-1)
