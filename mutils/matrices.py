#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix utility functions for sparse matrix operations.
"""

import torch


def spdiags(input):
    """
    Create a sparse diagonal matrix from a 1D tensor.

    Args:
        input (torch.Tensor): 1D tensor containing diagonal values

    Returns:
        torch.sparse.Tensor: Sparse diagonal matrix
    """
    length = input.shape[0]
    diag_indices = torch.arange(length).to(input.device)
    spdiag = torch.sparse_coo_tensor(
        torch.stack([diag_indices, diag_indices], dim=0),
        input,
        (length, length)
    )

    return spdiag


def block_diagonal(A, num_blocks=3):
    """
    Create a block diagonal matrix by replicating a matrix along the diagonal.

    This creates a larger matrix with copies of A along the diagonal, useful for
    applying the same linear operation to multiple coordinate dimensions.

    Args:
        A (torch.Tensor): Square sparse matrix of size (N, N)
        num_blocks (int, optional): Number of diagonal blocks. Defaults to 3.

    Returns:
        torch.sparse.Tensor: Block diagonal matrix of size (N*num_blocks, N*num_blocks)
    """
    N = A.size(0)
    assert A.size(0) == A.size(1), "Input matrix must be square"

    # Convert to sparse if needed
    if not A.is_sparse:
        A = A.to_sparse()

    indices = A.indices()
    values = A.values()

    new_indices = []
    new_values = []

    # Create each diagonal block
    for i in range(num_blocks):
        block_indices = indices.clone()
        # Shift indices for this block
        block_indices[0, :] += i * N
        block_indices[1, :] += i * N

        new_indices.append(block_indices)
        new_values.append(values)

    # Concatenate all blocks
    final_indices = torch.cat(new_indices, dim=1)
    final_values = torch.cat(new_values)

    # Create the final sparse matrix
    result = torch.sparse_coo_tensor(
        final_indices,
        final_values,
        size=(N * num_blocks, N * num_blocks)
    )

    return result.coalesce()

def assemble_Lagrange(A, B):
    """
    Create a saddle point matrix of form:
    [A  B^T]
    [B   0 ]

    Args:
        A: Sparse matrix of size (n×n)
        B: Sparse matrix of size (m×n)

    Returns:
        Assembled sparse matrix of size ((n+m)×(n+m))
    """
    # Get dimensions
    n = A.size(0)  # A is n×n
    m = B.size(0)  # B is m×n
    total_size = n + m

    # Extract sparse matrix components
    A_indices = A.indices()
    A_values = A.values()
    B_indices = B.indices()
    B_values = B.values()

    # Adjust B indices for the bottom-left block
    # Rows shift by n, columns stay the same
    B_indices_adjusted = torch.stack([
        B_indices[0] + n,  # Add n to row indices
        B_indices[1]       # Column indices stay the same
    ])

    # Create B^T indices for the top-right block
    # Transpose by swapping row and column indices, then shift columns by n
    BT_indices = torch.stack([
        B_indices[1],      # Row indices are B's column indices
        B_indices[0] + n   # Column indices are B's row indices + n
    ])

    # Concatenate all indices and values
    all_indices = torch.cat([
        A_indices,           # Top-left block (A)
        B_indices_adjusted,  # Bottom-left block (B)
        BT_indices,          # Top-right block (B^T)
    ], dim=1)

    all_values = torch.cat([
        A_values,  # Values from A
        B_values,  # Values from B
        B_values,  # Values from B^T (same as B values)
    ])

    # Create the assembled sparse matrix
    assembled = torch.sparse_coo_tensor(
        indices=all_indices,
        values=all_values,
        size=(total_size, total_size)
    )

    return assembled.coalesce()
