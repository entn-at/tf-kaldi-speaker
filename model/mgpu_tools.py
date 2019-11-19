import tensorflow as tf
import operator
import six


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


def assign_to_device(device, ps_device=None):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on.  Example values are GPU:0 and
        CPU:0.

    If ps_device is not set then the variables will be placed on the device.
    The best device for shared varibles depends on the platform as well as the
    model.  Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.

    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device
    return _assign


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        v = grad_and_vars[0][1]
        if len(grads) == 0:
            tf.logging.info("%s has no gradient, skip..." % v.op.name)
            continue

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class GpuParamServerDeviceSetter(object):
    """Used with tf.device() to place variables on the least loaded GPU.

        A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
        'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
        placed on the least loaded gpu. All other Ops, which will be the computation
        Ops, will be placed on the worker_device.
    """
    def __init__(self, worker_device, ps_devices):
        """Initializer for GpuParamServerDeviceSetter.
        Args:
            worker_device: the device to use for computation Ops.
            ps_devices: a list of devices to use for Variable Ops. Each variable is
            assigned to the least loaded device.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        # if op.device:
        #     return op.device
        # if op.type not in PS_OPS:
        #     return self.worker_device

        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op not in PS_OPS:
            return self.worker_device

        # Gets the least loaded ps_device
        device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size
        return device_name


def create_device_setter(is_cpu_ps, worker, num_gpus):
    """Create device setter object."""
    if is_cpu_ps:
        return "/cpu:0"
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return GpuParamServerDeviceSetter(worker, gpus)


def local_device_setter(num_devices,
                        ps_device_type,
                        worker_device,
                        ps_ops=None,
                        ps_strategy=None):
    from tensorflow.python.training import device_setter
    from tensorflow.python.framework import device as pydev
    from tensorflow.core.framework import node_def_pb2

    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()
    return _local_device_chooser
