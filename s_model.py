import keras

from keras import optimizers
from keras import objectives
from keras import metrics as metrics_module
from keras.models import Sequential, Model
from keras import backend as K

import copy

'''
 I redefine Model in order to avoid batch size normalization of the loss.
 This is because memory networks seems to work better without this.
 This way of doing it is kind of tricky. TODO: I should find a better way.
'''

def collect_metrics(metrics, output_names):
    if not metrics:
        return [[] for _ in output_names]
    if isinstance(metrics, list):
        # we then apply all metrics to all outputs.
        return [copy.copy(metrics) for _ in output_names]
    elif isinstance(metrics, dict):
        nested_metrics = []
        for name in output_names:
            output_metrics = metrics.get(name, [])
            if not isinstance(output_metrics, list):
                output_metrics = [output_metrics]
            nested_metrics.append(output_metrics)
        return nested_metrics
    else:
        raise TypeError('Type of `metrics` argument not understood. '
                        'Expected a list or dictionary, found: ' +
                        str(metrics))


def weighted_objective(fn):
    '''Transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.
    '''
    def weighted(y_true, y_pred, weights, mask=None):
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            mask = K.cast(mask, K.floatx())
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= K.mean(mask)

        # reduce score_array to same ndim as weight array
        ndim = K.ndim(score_array)
        weight_ndim = K.ndim(weights)
        score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))

        # apply sample weighting
        if weights is not None:
            score_array *= weights
            score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return K.sum(score_array)
    return weighted

class sModel(Model):

    # Redefine compile in order to avoid weighting of losses
    def compile(self, optimizer, loss, metrics=[], loss_weights=None,
        sample_weight_mode=None, **kwargs):
        #super(sModel, self).compile(optimizer, loss, metrics, loss_weights, 
        #                            sample_weights_mode, **kwargs)
        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode
        self.loss = loss
        self.loss_weights = loss_weights

        # prepare loss weights
        if loss_weights is None:
            loss_weights_list = [1. for _ in range(len(self.outputs))]
        elif isinstance(loss_weights, dict):
            for name in loss_weights:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss_weights '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_weights_list = []
            for name in self.output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif isinstance(loss_weights, list):
            if len(loss_weights) != len(self.outputs):
                raise ValueError('When passing a list as loss_weights, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss_weights=' +
                                 str(loss_weights))
            loss_weights_list = loss_weights
        else:
            raise TypeError('Could not interpret loss_weights argument: ' +
                            str(loss_weights) +
                            ' - expected a list of dicts.')

        # prepare loss functions
        if isinstance(loss, dict):
            for name in loss:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_functions = []
            for name in self.output_names:
                if name not in loss:
                    raise ValueError('Output "' + name +
                                     '" missing from loss dictionary.')
                loss_functions.append(objectives.get(loss[name]))
        elif isinstance(loss, list):
            if len(loss) != len(self.outputs):
                raise ValueError('When passing a list as loss, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss=' +
                                 str(loss))
            loss_functions = [objectives.get(l) for l in loss]
        else:
            loss_function = objectives.get(loss)
            loss_functions = [loss_function for _ in range(len(self.outputs))]
        self.loss_functions = loss_functions
        weighted_losses = [weighted_objective(fn) for fn in loss_functions]

        # prepare output masks
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        if not isinstance(masks, list):
            masks = [masks]

        # prepare sample weights
        if isinstance(sample_weight_mode, dict):
            for name in sample_weight_mode:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in '
                                     'sample_weight_mode dictionary: "' +
                                     name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            sample_weights = []
            sample_weight_modes = []
            for name in self.output_names:
                if name not in sample_weight_mode:
                    raise ValueError('Output "' + name +
                                     '" missing from sample_weight_modes '
                                     'dictionary')
                if sample_weight_mode.get(name) == 'temporal':
                    weight = K.placeholder(ndim=2,
                                           name=name + '_sample_weights')
                    sample_weight_modes.append('temporal')
                else:
                    weight = K.placeholder(ndim=1,
                                           name=name + '_sample_weights')
                    sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif isinstance(sample_weight_mode, list):
            if len(sample_weight_mode) != len(self.outputs):
                raise ValueError('When passing a list as sample_weight_mode, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed '
                                 'sample_weight_mode=' +
                                 str(sample_weight_mode))
            sample_weights = []
            sample_weight_modes = []
            for mode, name in zip(sample_weight_mode, self.output_names):
                if mode == 'temporal':
                    weight = K.placeholder(ndim=2,
                                           name=name + '_sample_weights')
                    sample_weight_modes.append('temporal')
                else:
                    weight = K.placeholder(ndim=1,
                                           name=name + '_sample_weights')
                    sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            if sample_weight_mode == 'temporal':
                sample_weights = [K.placeholder(ndim=2,
                                                name=name + '_sample_weights')
                                  for name in self.output_names]
                sample_weight_modes = ['temporal'
                                       for name in self.output_names]
            else:
                sample_weights = [K.placeholder(ndim=1,
                                                name=name + '_sample_weights')
                                  for name in self.output_names]
                sample_weight_modes = [None for name in self.output_names]
        self.sample_weight_modes = sample_weight_modes

        # prepare targets of model
        self.targets = []
        for i in range(len(self.outputs)):
            shape = self.internal_output_shapes[i]
            name = self.output_names[i]
            self.targets.append(K.placeholder(ndim=len(shape),
                                              name=name + '_target',
                                              sparse=K.is_sparse(self.outputs[i]),
                                              dtype=K.dtype(self.outputs[i])))

        # prepare metrics
        self.metrics = metrics
        self.metrics_names = ['loss']
        self.metrics_tensors = []

        # compute total loss
        total_loss = None
        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            weighted_loss = weighted_losses[i]
            sample_weight = sample_weights[i]
            mask = masks[i]
            loss_weight = loss_weights_list[i]
            output_loss = weighted_loss(y_true, y_pred,
                                        sample_weight, mask)
            if len(self.outputs) > 1:
                self.metrics_tensors.append(output_loss)
                self.metrics_names.append(self.output_names[i] + '_loss')
            if total_loss is None:
                total_loss = loss_weight * output_loss
            else:
                total_loss += loss_weight * output_loss

        # add regularization penalties
        # and other layer-specific losses
        for loss_tensor in self.losses:
            total_loss += loss_tensor

        # list of same size as output_names.
        # contains tuples (metrics for output, names of metrics)
        nested_metrics = collect_metrics(metrics, self.output_names)

        def append_metric(layer_num, metric_name, metric_tensor):
            """Helper function, used in loop below"""
            if len(self.output_names) > 1:
                metric_name = self.output_layers[layer_num].name + '_' + metric_name

            self.metrics_names.append(metric_name)
            self.metrics_tensors.append(metric_tensor)

        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            output_metrics = nested_metrics[i]

            for metric in output_metrics:
                if metric == 'accuracy' or metric == 'acc':
                    # custom handling of accuracy
                    # (because of class mode duality)
                    output_shape = self.internal_output_shapes[i]
                    acc_fn = None
                    if output_shape[-1] == 1 or self.loss_functions[i] == objectives.binary_crossentropy:
                        # case: binary accuracy
                        acc_fn = metrics_module.binary_accuracy
                    elif self.loss_functions[i] == objectives.sparse_categorical_crossentropy:
                        # case: categorical accuracy with sparse targets
                        acc_fn = metrics_module.sparse_categorical_accuracy
                    else:
                        acc_fn = metrics_module.categorical_accuracy

                    append_metric(i, 'acc', acc_fn(y_true, y_pred))
                else:
                    metric_fn = metrics_module.get(metric)
                    metric_result = metric_fn(y_true, y_pred)

                    if not isinstance(metric_result, dict):
                        metric_result = {
                            metric_fn.__name__: metric_result
                        }

                    for name, tensor in six.iteritems(metric_result):
                        append_metric(i, name, tensor)

        # prepare gradient updates and state updates
        self.total_loss = total_loss
        self.sample_weights = sample_weights

        # functions for train, test and predict will
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self._function_kwargs = kwargs

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # collected trainable weights and sort them deterministically.
        trainable_weights = self.trainable_weights
        # Sort weights by name
        if trainable_weights:
            if K.backend() == 'theano':
                trainable_weights.sort(key=lambda x: x.name if x.name else x.auto_name)
            else:
                trainable_weights.sort(key=lambda x: x.name)
        self._collected_trainable_weights = trainable_weights
