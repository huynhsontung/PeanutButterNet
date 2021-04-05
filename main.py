from PeanutButterNet.utils import extract_kernel_sizes
from enum import Enum
import tensorflow as tf


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


class HParams():
    def __init__(self, **kwargs) -> None:
        self.params = kwargs
        # stem_multiplier=1.0,
        # dense_dropout_keep_prob=0.5,
        # num_cells=12,
        # filter_scaling_rate=2.0,
        # drop_path_keep_prob=1.0,
        # num_conv_filters=44,
        # use_aux_head=1,
        # num_reduction_layers=2,
        # data_format='NHWC',
        # skip_reduction_layer_input=0,
        # total_training_steps=250000,
        # use_bounded_activation=False,


class Operation():
    def __init__(self, op_type: OperationType, hparams: HParams,
                 reduce=False, **kwargs) -> None:
        op_name = op_type.name.lower()
        self.ops: list[tf.keras.layers.Layer] = []
        if op_type == OperationType.IDENTITY:
            pass
        elif "conv" in op_name:
            filter_size = hparams["num_conv_filters"]
            stride = 2 if reduce else 1
            kernel_sizes = extract_kernel_sizes(op_type.name)
            dilation_rate = 2 if "dilated" in op_name else 1
            if "sep" in op_name:
                self.ops = [
                    tf.keras.layers.SeparableConv2D(
                        filter_size,
                        kernel,
                        stride,
                        dilation_rate=dilation_rate, **kwargs) for kernel in kernel_sizes]
            else:
                self.ops = [
                    tf.keras.layers.Conv2D(
                        filter_size,
                        kernel,
                        stride,
                        dilation_rate=dilation_rate, **kwargs) for kernel in kernel_sizes]
        elif "max_pool" in op_name:
            # TODO: Implement max pool
            pass
        elif "avg_pool" in op_name:
            # TODO: Implement avg pool
            pass

    def __call__(self, hidden: tf.Tensor) -> tf.Tensor:
        for op in self.ops:
            hidden = op(hidden)
        
        return hidden


class Block():
    def __init__(self, op0: Operation, op1: Operation) -> None:
        self.op0 = op0
        self.op1 = op1

    def __call__(self, hidden0: tf.Tensor, hidden1: tf.Tensor, concat=False) -> tf.Tensor:
        hidden0 = self.op0(hidden0)
        hidden1 = self.op1(hidden1)
        if concat:
            return tf.keras.layers.concatenate([hidden0, hidden1], axis=-1)
        else:
            return tf.keras.layers.add([hidden0, hidden1])