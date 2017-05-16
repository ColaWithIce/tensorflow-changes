

@add_arg_scope
def convolution3d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer,
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
  """Adds a 2D convolution followed by an optional batch_norm layer.

  `convolution2d` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved with the `inputs` to produce a
  `Tensor` of activations. If a `normalizer_fn` is provided (such as
  `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
  None and a `biases_initializer` is provided then a `biases` variable would be
  created and added the activations. Finally, if `activation_fn` is not `None`,
  it is applied to the activations as well.

  Performs a'trous convolution with input stride equal to rate if rate is
  greater than one.

  Args:
    inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
    num_outputs: integer, the number of output filters.
    kernel_size: a list of length 2 `[kernel_height, kernel_width]` of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2 `[stride_height, stride_width]`.
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: one of `VALID` or `SAME`.
    rate: integer. If less than or equal to 1, a standard convolution is used.
      If greater than 1, than the a'trous convolution is applied and `stride`
      must be set to 1.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionay containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    a tensor representing the output of the operation.

  Raises:
    ValueError: if both 'rate' and `stride` are larger than one.
  """
  with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w, kernel_d = utils.three_element_tuple(kernel_size)
    stride_h, stride_w, stride_d = utils.three_element_tuple(stride)
    if rate > 1 and (stride_h > 1 or stride_w > 1, stride_d > 1):
      raise ValueError('Only one of rate or stride can be larger than one')
    num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=5)
    weights_shape = [kernel_h, kernel_w, kernel_d,
                     num_filters_in, num_outputs]
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')
    weights = variables.model_variable('weights',
                                       shape=weights_shape,
                                       dtype=dtype,
                                       initializer=weights_initializer,
                                       regularizer=weights_regularizer,
                                       collections=weights_collections,
                                       trainable=trainable)
    if rate > 1:
      outputs = nn.atrous_conv2d(inputs, weights, rate, padding=padding)
    else:
      outputs = nn.conv3d(inputs, weights, [1, stride_h, stride_w, stride_d, 1],
                          padding=padding)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections,
                                          trainable=trainable)
        outputs = nn.bias_add(outputs, biases)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def convolution3d_transpose(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    padding='SAME',
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer,
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a convolution2d_transpose with an optional batch normalization layer.

  The function creates a variable called `weights`, representing the
  kernel, that is convolved with the input. If `batch_norm_params` is `None`, a
  second variable called 'biases' is added to the result of the operation.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_outputs: integer, the number of output filters.
    kernel_size: a list of length 2 holding the [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionay containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    a tensor representing the output of the operation.

  Raises:
    ValueError: if 'kernel_size' is not a list of length 2.
  """
  with variable_scope.variable_scope(
      scope, 'Conv3d_transpose', [inputs], reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w, kernel_d = utils.three_element_tuple(kernel_size)
    stride_h, stride_w, stride_d = utils.three_element_tuple(stride)
    num_filters_in = utils.last_dimension(
        inputs.get_shape(), min_rank=5)
    weights_shape = [kernel_h, kernel_w, kernel_d, num_outputs, num_filters_in]
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')
    weights = variables.model_variable(
        'weights',
        shape=weights_shape,
        dtype=dtype,
        initializer=weights_initializer,
        regularizer=weights_regularizer,
        trainable=trainable,
        collections=weights_collections)

    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    height, width, depth = inputs_shape[1], inputs_shape[2], inputs_shape[3]

    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
      if isinstance(dim_size, ops.Tensor):
        dim_size = math_ops.mul(dim_size, stride_size)
      elif dim_size is not None:
        dim_size *= stride_size

      if padding == 'VALID' and dim_size is not None:
        dim_size += max(kernel_size - stride_size, 0)
      return dim_size

    # Infer the dynamic output shape:
    out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
    out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
    out_depth = get_deconv_dim(depth, stride_d, kernel_d, padding)

    output_shape = array_ops.pack(
        [batch_size, out_height, out_width, out_depth, num_outputs])
    outputs = nn.conv3d_transpose(inputs, weights, output_shape,
                                  [1, stride_h, stride_w, stride_d, 1],
                                  padding=padding)

    # Infer the static output shape:
    out_shape = inputs.get_shape().as_list()
    out_shape[-1] = num_outputs
    out_shape[1] = get_deconv_dim(out_shape[1], stride_h, kernel_h, padding)
    out_shape[2] = get_deconv_dim(out_shape[2], stride_w, kernel_w, padding)
    out_shape[3] = get_deconv_dim(out_shape[3], stride_d, kernel_d, padding)
    outputs.set_shape(out_shape)

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections)
        outputs = nn.bias_add(outputs, biases)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


