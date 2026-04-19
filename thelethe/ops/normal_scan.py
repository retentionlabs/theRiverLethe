"""
Enhanced Associative Scan for Neural Memory Operations (Eager Mode)
Optimized implementation with robust PyTree handling and memory efficiency
"""
from typing import Callable, Optional
import warnings

import torch
import torch.utils._pytree as pytree
from torch.utils.checkpoint import checkpoint
from torch._higher_order_ops import scan  # for future use


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
    checkpoint_group: Optional[int] = None
) -> tuple[pytree.PyTree, pytree.PyTree]:
    """
    Performs an inclusive scan with a combine function.

    IMPORTANT: This implementation is torch.compile compatible and only supports
    simple PyTree structures (tensors, lists/tuples of tensors, dicts of tensors).
    Complex nested structures are not supported due to torch.compile limitations.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> (Tensor, Tensor)``,
            or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor or pytree with tensor leaves): The initial scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
            of ``combine_fn``.
        xs (torch.Tensor or pytree with tensor leaves): The input tensor, or nested pytree of tensors.

    Kwargs:
        dim (int): the dimension to scan over, default 0.
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        checkpoint_group (Optional[int]): Number of checkpoint groups. If None or negative, defaults to 0 (no checkpointing).
            Note: This parameter will not be available when using PyTorch's official scan implementation.

    Returns:
        final_carry (torch.Tensor or pytree with tensor leaves),
            the final carry of the scan operation with same pytree structure as init.
        out (torch.Tensor or pytree with tensor leaves),
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.

    Raises:
        TypeError: If xs is not compatible with this implementation.

    Warnings:
        FutureWarning: The 'checkpoint_group' parameter is a custom extension and will not be
            available when this implementation is replaced by PyTorch's official scan.
    """
    carry = init

    # Handle checkpoint_group parameter and emit warnings
    if checkpoint_group is None or checkpoint_group < 0:
        checkpoint_group = 0
    elif checkpoint_group > 0:
        warnings.warn(
            "The 'checkpoint_group' parameter is a custom extension and will not be "
            "available when this implementation is replaced by PyTorch's official scan. "
            "Consider using torch.utils.checkpoint.checkpoint directly for future compatibility.",
            FutureWarning,
            stacklevel=2
        )

    # Determine number of items and initialize output structure
    if isinstance(xs, torch.Tensor):
        shape = xs.shape
        num_items = shape[dim]
        out = torch.empty(shape, dtype=xs.dtype, device=xs.device)
        if dim == 0:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = xs[i]
                    current_carry, y = combine_fn(current_carry, x)
                    out[i] = y
                return current_carry
        elif dim == 1:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = xs[:, i]
                    current_carry, y = combine_fn(current_carry, x)
                    out[:, i] = y
                return current_carry
        elif dim == 2:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = xs[:, :, i]
                    current_carry, y = combine_fn(current_carry, x)
                    out[:, :, i] = y
                return current_carry
        else:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = xs.select(dim, i)
                    current_carry, y = combine_fn(current_carry, x)
                    out.select(dim, i).copy_(y)
                return current_carry
    elif isinstance(xs, dict):
        shape = next(iter(xs.values())).shape
        num_items = shape[dim]
        out = {key: torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for key, tensor in xs.items()}
        dict_keys = list(xs.keys())
        out_tensors = [out[key] for key in dict_keys]
        if dim == 0:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = {key: tensor[i] for key, tensor in xs.items()}
                    current_carry, y = combine_fn(current_carry, x)
                    for j, key in enumerate(dict_keys):
                        out_tensors[j][i] = y[key]
                return current_carry
        elif dim == 1:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = {key: tensor[:, i] for key, tensor in xs.items()}
                    current_carry, y = combine_fn(current_carry, x)
                    for j, key in enumerate(dict_keys):
                        out_tensors[j][:, i] = y[key]
                return current_carry
        elif dim == 2:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = {key: tensor[:, :, i] for key, tensor in xs.items()}
                    current_carry, y = combine_fn(current_carry, x)
                    for j, key in enumerate(dict_keys):
                        out_tensors[j][:, :, i] = y[key]
                return current_carry
        else:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x = {key: tensor.select(dim, i) for key, tensor in xs.items()}
                    current_carry, y = combine_fn(current_carry, x)
                    for j, key in enumerate(dict_keys):
                        out_tensors[j].select(dim, i).copy_(y[key])
                return current_carry
    elif isinstance(xs, (list, tuple)):
        first_tensor = xs[0]
        num_items = first_tensor.shape[dim]
        out_tensors = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for tensor in xs]
        out = tuple(out_tensors) if isinstance(xs, tuple) else out_tensors
        if dim == 0:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x_list = [t[i] for t in xs]
                    current_carry, y = combine_fn(current_carry, x_list)
                    for j, value in enumerate(y):
                        out[j][i] = value
                return current_carry
        elif dim == 1:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x_list = [t[:, i] for t in xs]
                    current_carry, y = combine_fn(current_carry, x_list)
                    for j, value in enumerate(y):
                        out[j][:, i] = value
                return current_carry
        elif dim == 2:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x_list = [t[:, :, i] for t in xs]
                    current_carry, y = combine_fn(current_carry, x_list)
                    for j, value in enumerate(y):
                        out[j][:, :, i] = value
                return current_carry
        else:
            def scan_fn(current_carry, i_start, i_end):
                indices = range(i_end - 1, i_start - 1, -1) if reverse else range(i_start, i_end)
                for i in indices:
                    x_list = [t.select(dim, i) for t in xs]
                    current_carry, y = combine_fn(current_carry, x_list)
                    for j, value in enumerate(y):
                        out[j].select(dim, i).copy_(value)
                return current_carry
    else:
        raise TypeError(f"Unsupported input type: {type(xs)}")

    # Execute scan with or without checkpointing
    if checkpoint_group > 0:
        ckpt_every_n = max(1, num_items // checkpoint_group)
        for k in range(0, num_items, ckpt_every_n):
            carry = checkpoint(scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False)
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


# Scan function is recommended to be compiled for better performance
compiled_scan = torch.compile(scan, mode="max-autotune")
