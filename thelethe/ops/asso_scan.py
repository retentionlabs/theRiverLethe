"""
Enhanced Scan for Neural Memory Operations (Eager Mode)
Optimized implementation with robust PyTree handling and memory efficiency
"""
from typing import Callable, Optional

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops import associative_scan  # for future use


def _slice_tensor(
    tensor: torch.Tensor, start: int, end: Optional[int] = None,
    stride: int = 1, axis: int = 0
) -> torch.Tensor:
    """Create optimized tensor slice along specified axis"""
    # Fast paths for common dimensions
    if axis == 0:
        return tensor[start:end:stride]
    elif axis == 1:
        return tensor[:, start:end:stride]
    elif axis == 2:
        return tensor[:, :, start:end:stride]
    else:
        # General case - create slice tuple for arbitrary dimensions
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start, end, stride)
        return tensor[tuple(slices)]


def _merge_sequences(
    even_seq: torch.Tensor, odd_seq: torch.Tensor, axis: int = 0
) -> torch.Tensor:
    """Efficiently merge even and odd sequences with proper padding"""
    even_len = even_seq.shape[axis]
    odd_len = odd_seq.shape[axis]

    # Handle different sequence lengths
    if even_len == odd_len:
        # Same length: direct interleaving
        combined = torch.stack([even_seq, odd_seq], dim=axis + 1)
        merged = torch.flatten(combined, start_dim=axis, end_dim=axis + 1)
    elif even_len == odd_len + 1:
        # Even sequence is longer by 1
        if odd_len == 0:
            return even_seq

        # Use optimized _slice_tensor
        even_part = _slice_tensor(even_seq, 0, odd_len, 1, axis)
        last_even = _slice_tensor(even_seq, odd_len, None, 1, axis)

        combined = torch.stack([even_part, odd_seq], dim=axis + 1)
        interleaved = torch.flatten(combined, start_dim=axis, end_dim=axis + 1)
        merged = torch.cat([interleaved, last_even], dim=axis)

    elif odd_len == even_len + 1:
        # Odd sequence is longer by 1
        if even_len == 0:
            return odd_seq

        # Use optimized _slice_tensor
        odd_part = _slice_tensor(odd_seq, 0, even_len, 1, axis)
        last_odd = _slice_tensor(odd_seq, even_len, None, 1, axis)

        combined = torch.stack([even_seq, odd_part], dim=axis + 1)
        interleaved = torch.flatten(combined, start_dim=axis, end_dim=axis + 1)
        merged = torch.cat([interleaved, last_odd], dim=axis)
    else:
        # Unexpected length difference
        raise ValueError(
            f"Cannot merge sequences with lengths {even_len} and {odd_len}. "
            f"Length difference should be at most 1."
        )

    return merged


def _validate_inputs(tensors: list, axis: int) -> int:
    """Validate tensor inputs and return sequence length"""
    if not tensors:
        return 0

    ref_tensor = tensors[0]
    seq_length = ref_tensor.shape[axis]

    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape[axis] != seq_length:
            raise ValueError(
                f"Tensor dimension mismatch at axis {axis}: "
                f"tensor[0].shape={ref_tensor.shape}, "
                f"tensor[{i}].shape={tensor.shape}"
            )

    return seq_length


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int = 0,
    reverse: bool = False,
    combine_mode: str = "generic"  # no needed, kept for compatibility
) -> pytree.PyTree:
    """
    Perform parallel associative scan using divide-and-conquer strategy

    Enhanced implementation for neural memory (for TTT, TITANS+) with:
    - Robust PyTree handling via dm-tree
    - Optimized tensor operations
    - Memory-efficient processing
    - Enhanced error reporting

    Args:
        combine_fn: Associative binary operator (a, b) -> c
        xs: Input data (tensors, dicts, lists, or nested structures)
        dim: Axis dimension for scanning (default: 0)
        reverse: Process sequence in reverse order
        combine_mode: Mode for combining tensors ("generic" only, keep this argument for compatibility with pytorch associative_scan proto API)

    Returns:
        Scanned sequence with same structure as input

    Raises:
        TypeError: If combine_fn is not callable
        ValueError: If tensor dimensions don't match
        ImportError: If dm-tree is not available
    """
    # Input validation
    if not callable(combine_fn):
        raise TypeError(
            f"combine_fn must be callable, got {type(combine_fn).__name__}"
        )

    # Handle edge cases
    if xs is None:
        return None

    # Fast path: bypass PyTree processing for simple tensor cases
    is_simple_tensor = False
    is_simple_case = True
    if isinstance(xs, torch.Tensor):
        # Single tensor input - no PyTree overhead
        flat_tensors = [xs]
        is_simple_tensor = True
    elif isinstance(xs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in xs):
        # Tensor list/tuple input - bypass flatten/unflatten
        flat_tensors = list(xs)
        original_type = type(xs)
    else:
        # Complex PyTree structure - use dm-tree only when necessary
        try:
            import tree
        except ImportError:
            raise ImportError("dm-tree library required. Install with: pip install dm-tree")

        is_simple_case = False
        try:  # Flatten nested structure using dm-tree
            flat_tensors = tree.flatten(xs)
            sequence_template = xs
        except Exception as e:
            raise ValueError(f"Failed to flatten input structure: {e}")

    if not flat_tensors:
        if isinstance(xs, (list, tuple)):
            return type(xs)([])
        elif isinstance(xs, dict):
            return {}
        else:
            return xs

    # Apply reverse transformation
    if reverse:
        flat_tensors = [torch.flip(t, [dim]) for t in flat_tensors]

    # Validate tensor compatibility
    _validate_inputs(flat_tensors, dim)

    def _combine_flat_tensors(left_tensors: list, right_tensors: list) -> list:
        """Apply combine function to flattened tensor lists"""
        if is_simple_tensor:  # Single tensor case
            combined_result = combine_fn(left_tensors[0], right_tensors[0])
            return [combined_result]
        elif is_simple_case:  # Simple list/tuple case
            left_structured = original_type(left_tensors)
            right_structured = original_type(right_tensors)
            combined_result = combine_fn(left_structured, right_structured)
            return list(combined_result)
        else:  # Complex PyTree case
            # Reconstruct original structure for operator
            left_structured = tree.unflatten_as(sequence_template, left_tensors)
            right_structured = tree.unflatten_as(sequence_template, right_tensors)

            # Apply user-defined operator
            combined_result = combine_fn(left_structured, right_structured)

            # Flatten result back to list
            return tree.flatten(combined_result)

    # Core recursive scanning algorithm
    def _recursive_scan(tensor_list: list) -> list:
        """
        Divide-and-conquer associative scan implementation

        Algorithm:
        1. Base case: sequences of length < 2
        2. Pair-wise combine adjacent elements
        3. Recursively scan reduced sequence
        4. Compute final results and merge
        """
        current_length = tensor_list[0].shape[dim]

        # Base case: trivial sequences
        if current_length < 2:
            return tensor_list

        # Step 1: Combine adjacent pairs (downward pass)
        left_elements = [_slice_tensor(t, 0, -1, 2, dim) for t in tensor_list]
        right_elements = [_slice_tensor(t, 1, None, 2, dim) for t in tensor_list]

        paired_results = _combine_flat_tensors(left_elements, right_elements)

        # Step 2: Recursive call on reduced problem
        scanned_pairs = _recursive_scan(paired_results)

        # Step 3: Compute even-indexed results (upward pass)
        if current_length % 2 == 0:
            # Even length: use prefix of scanned pairs
            prefix_pairs = [_slice_tensor(t, 0, -1, 1, dim) for t in scanned_pairs]
            suffix_elements = [_slice_tensor(t, 2, None, 2, dim) for t in tensor_list]
            even_results = _combine_flat_tensors(prefix_pairs, suffix_elements)
        else:
            # Odd length: use full-scanned pairs
            suffix_elements = [_slice_tensor(t, 2, None, 2, dim) for t in tensor_list]
            even_results = _combine_flat_tensors(scanned_pairs, suffix_elements)

        # Step 4: Prepare final even sequence with identity element
        identity_elements = [_slice_tensor(t, 0, 1, 1, dim) for t in tensor_list]

        complete_even = []
        for identity_elem, even_elem in zip(identity_elements, even_results):
            if even_elem.numel() > 0 and identity_elem.shape[dim] > 0:
                complete_even.append(torch.cat([identity_elem, even_elem], dim=dim))
            elif even_elem.numel() > 0:
                complete_even.append(even_elem)
            else:
                complete_even.append(identity_elem)

        # Step 5: Merge even and odd sequences
        final_results = []
        for even_seq, odd_seq in zip(complete_even, scanned_pairs):
            try:
                merged = _merge_sequences(even_seq, odd_seq, dim)
                final_results.append(merged)
            except ValueError as e:
                # Debug information for merge issues
                print(f"Merge error: {e}")
                print(f"Even shape: {even_seq.shape}, Odd shape: {odd_seq.shape}")
                print(f"Original length: {current_length}")
                raise

        return final_results

    # Execute main algorithm
    scanned_tensors = _recursive_scan(flat_tensors)

    # Undo reverse transformation
    if reverse:
        scanned_tensors = [torch.flip(t, [dim]) for t in scanned_tensors]

    # Reconstruct original nested structure
    if is_simple_tensor:
        return scanned_tensors[0]
    elif is_simple_case:
        return original_type(scanned_tensors)
    else:
        return tree.unflatten_as(sequence_template, scanned_tensors)
