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
    def __init__(self, op_type: OperationType, reduce=False,
                 num_conv_filters=32, **kwargs) -> None:
        op_name = op_type.name.lower()
        self.ops: list[tf.keras.layers.Layer] = []
        if op_type == OperationType.IDENTITY:
            pass
        else:
            stride = 2 if reduce else 1
            kernel_sizes = extract_kernel_sizes(op_type.name)
            dilation_rate = 2 if "dilated" in op_name else 1
            if "conv" in op_name:
                if "sep" in op_name:
                    self.ops = [
                        keras.layers.SeparableConv2D(
                            num_conv_filters,
                            kernel,
                            stride,
                            dilation_rate=dilation_rate, **kwargs) for kernel in kernel_sizes]
                else:
                    self.ops = [
                        keras.layers.Conv2D(
                            num_conv_filters,
                            kernel,
                            stride,
                            dilation_rate=dilation_rate, **kwargs) for kernel in kernel_sizes]
            elif "max_pool" in op_name:
                self.ops = [
                    keras.layers.MaxPool2D(
                        pool_size=pool_size,
                        strides=stride) for pool_size in kernel_sizes]
            elif "avg_pool" in op_name:
                self.ops = [
                    keras.layers.AveragePooling2D(
                        pool_size=pool_size,
                        strides=stride) for pool_size in kernel_sizes]
            else:
                raise ValueError("Operation type not supported")

    def __call__(self, hidden: tf.Tensor) -> tf.Tensor:
        for op in self.ops:
            hidden = op(hidden)

        return hidden


class Block():
    def __init__(self, op0: Operation, op1: Operation, concat=False) -> None:
        self.op0 = op0
        self.op1 = op1
        self.concat = concat

    def __call__(self, hidden0: tf.Tensor, hidden1: tf.Tensor) -> tf.Tensor:
        hidden0 = self.op0(hidden0)
        hidden1 = self.op1(hidden1)
        if self.concat:
            return keras.layers.concatenate([hidden0, hidden1], axis=-1)
        else:
            return keras.layers.add([hidden0, hidden1])


class Cell():
    def __init__(self, blocks: list[Block],
                 connections: list[tuple[int, int]]) -> None:
        assert len(blocks) == len(connections)
        self.blocks = blocks
        self.connections = connections

    def __call__(self, hidden0: tf.Tensor, hidden1: tf.Tensor) -> tf.Tensor:
        counter = 1
        connections = self.connections.copy()
        hiddens = [hidden0, hidden1]
        outputs: list[tf.Tensor] = []
        while connections:
            candidate_idx = [all([x <= counter for x in c])
                             for c in connections].index(True)
            block = self.blocks[candidate_idx]
            block_input = connections[candidate_idx]
            entry = block(hiddens[block_input[0]], hiddens[block_input[1]])

            connections.pop(candidate_idx)
            hiddens.append(entry)
            outputs.append(entry)
            counter += 1

        return keras.layers.concatenate(outputs, axis=-1)
