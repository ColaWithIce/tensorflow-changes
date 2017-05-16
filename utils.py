def three_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a `TensorShape`.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 3:
      raise ValueError('Must be a list with 3 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1]), int(int_or_tuple[2])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tensor_shape.TensorShape):
    if len(int_or_tuple) == 3:
      return int_or_tuple[0], int_or_tuple[1], int_or_tuple[2]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 3')
