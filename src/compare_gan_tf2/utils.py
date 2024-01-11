# Copyright 2023 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/google/compare_gan/blob/master/compare_gan/utils.py
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from absl import logging
import six
import inspect
import functools
import collections

# In Python 2 the inspect module does not have FullArgSpec. Define a named tuple
# instead.
if hasattr(inspect, "FullArgSpec"):
    _FullArgSpec = inspect.FullArgSpec  # pylint: disable=invalid-name
else:
    _FullArgSpec = collections.namedtuple("FullArgSpec", [
        "args", "varargs", "varkw", "defaults", "kwonlyargs", "kwonlydefaults",
        "annotations"
    ])


def _getfullargspec(fn):
    """Python 2/3 compatible version of the inspect.getfullargspec method.

    Args:
      fn: The function object.

    Returns:
      A FullArgSpec. For Python 2 this is emulated by a named tuple.
    """
    arg_spec_fn = inspect.getfullargspec if six.PY3 else inspect.getargspec
    try:
        arg_spec = arg_spec_fn(fn)
    except TypeError:
        # `fn` might be a callable object.
        arg_spec = arg_spec_fn(fn.__call__)
    if six.PY3:
        assert isinstance(arg_spec, _FullArgSpec)
        return arg_spec
    return _FullArgSpec(
        args=arg_spec.args,
        varargs=arg_spec.varargs,
        varkw=arg_spec.keywords,
        defaults=arg_spec.defaults,
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})


def _has_arg(fn, arg_name):
    """Returns True if `arg_name` might be a valid parameter for `fn`.

    Specifically, this means that `fn` either has a parameter named
    `arg_name`, or has a `**kwargs` parameter.

    Args:
      fn: The function to check.
      arg_name: The name fo the parameter.

    Returns:
      Whether `arg_name` might be a valid argument of `fn`.
    """
    while isinstance(fn, functools.partial):
        fn = fn.func
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    arg_spec = _getfullargspec(fn)
    if arg_spec.varkw:
        return True
    return arg_name in arg_spec.args or arg_name in arg_spec.kwonlyargs


def call_with_accepted_args(fn, **kwargs):
    """Calls `fn` only with the keyword arguments that `fn` accepts."""
    kwargs = {k: v for k, v in six.iteritems(kwargs) if _has_arg(fn, k)}
    logging.debug("Calling %s with args %s.", fn, kwargs)
    return fn(**kwargs)
