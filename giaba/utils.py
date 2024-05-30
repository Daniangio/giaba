import numpy as np
from numba import jit

def split_reminder(x: np.ndarray, chunk_size: int, axis=0):
    indices = np.arange(chunk_size, x.shape[axis], chunk_size)
    return np.array_split(x, indices, axis)

def move_elements(arr, consecutive_idcs: np.ndarray, new_place_id: int):
    assert np.all(consecutive_idcs[1:] - consecutive_idcs[:-1] == 1)
    move_left = new_place_id < consecutive_idcs[0]
    
    tomove = np.array(arr[consecutive_idcs])
    if move_left:
        tobemoved = np.array(arr[new_place_id:consecutive_idcs[0]])
        arr[new_place_id:new_place_id+len(consecutive_idcs)] = tomove
        arr[new_place_id+len(consecutive_idcs):consecutive_idcs[-1] + 1] = tobemoved
    else:
        tobemoved = np.array(arr[consecutive_idcs[-1] + 1:new_place_id])
        arr[new_place_id-len(consecutive_idcs):new_place_id] = tomove
        arr[consecutive_idcs[0]:new_place_id-len(consecutive_idcs)] = tobemoved

    return arr

def stack_and_permute_elements(arr, seq_src, seq_trg):
    stacked_arr = np.copy(arr)
    if seq_src[0] < seq_trg[0]:
        stacked_arr = move_elements(stacked_arr, seq_src, seq_trg[0])
        id_src = seq_trg[0] - len(seq_src)
        id_trg = seq_trg[0]
        swap_arr = np.copy(stacked_arr)
        swap_arr[id_src], swap_arr[id_trg] = stacked_arr[id_trg], stacked_arr[id_src]
    else:
        stacked_arr = move_elements(stacked_arr, seq_src, seq_trg[-1] + 1)
        id_src = seq_trg[0] + len(seq_trg)
        id_trg = seq_trg[0]
        swap_arr = np.copy(stacked_arr)
        swap_arr[id_src], swap_arr[id_trg] = stacked_arr[id_trg], stacked_arr[id_src]
    return [stacked_arr, swap_arr]

def stack_consecutive_types(arr, type):
    all_arr = []
    type_indices = np.where(arr[:, 0] == type)[0]
    consecutive_sequences = np.split(type_indices, np.where(np.diff(type_indices) != 1)[0] + 1)
    if len(consecutive_sequences) > 1:
        for sequence_prev, sequence_next in zip(consecutive_sequences[:-1], consecutive_sequences[1:]):
            all_arr.extend(stack_and_permute_elements(arr, sequence_prev, sequence_next))
            all_arr.extend(stack_and_permute_elements(arr, sequence_next, sequence_prev))

    return all_arr

def first_n_occurrences(arr, n):
    unique_vals, counts = np.unique(arr, return_counts=True)
    result = []
    for val, count in zip(unique_vals, counts):
        indices = np.where(arr == val)[0]
        result.extend(indices[:min(n, count)])
    return result

def find_nth_smallest(arr, n):
    if n > len(arr):
        n = len(arr)
    return np.partition(arr, n-1)[n-1]

def swap_combs(seq_src, seq_trg):
    for i in range(len(seq_trg)):
        for j in range(len(seq_src)):
            yield seq_src[:j+1], seq_trg[:i+1]

def shift_chunk(arr: np.ndarray, chunk_idcs: np.ndarray, shift: int):
    # We spare computation time by not checking if chunk_idcs are consecutive.
    # We assume they are.
    shifted_arr = np.copy(arr)
    if shift < 0:
        shift = max(-chunk_idcs[0], shift)
        id_from = chunk_idcs[0] + shift
        id_to = chunk_idcs[-1] + 1
    else:
        shift = min(len(arr) - chunk_idcs[-1] - 1, shift)
        id_from = chunk_idcs[0]
        id_to = chunk_idcs[-1] + 1 + shift
    shifted_arr[id_from:id_to] = np.roll(shifted_arr[id_from:id_to], shift, axis=0)
    return shifted_arr

@jit(nopython=True)
def find_first(item: bool, vec: np.ndarray[bool]):
    """return the index of the first occurence of item in vec"""
    for i, value in enumerate(vec):
        if item == value:
            return i
    return -1