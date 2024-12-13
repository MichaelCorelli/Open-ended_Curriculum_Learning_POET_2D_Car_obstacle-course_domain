import numpy as np
import torch

def state_dict_to_vector(state_dict):
    params = []
    for key, value in state_dict.items():
        #print(f"Processing key: {key}, shape: {value.shape}, numel: {value.numel()}")
        np_value = value.detach().cpu().numpy().ravel()
        params.append(np_value)
    vector = np.concatenate(params)
    #print(f"Final vector length: {len(vector)}")
    return vector

def vector_to_state_dict(vector, reference_state_dict):
    vector = np.asarray(vector).flatten()  
   #print(f"Vector shape after flatten: {vector.shape}")
    new_state_dict = {}
    idx = 0

    for key, value in reference_state_dict.items():
        length = value.numel()
        np_values = vector[idx:idx + length]

        if len(np_values) != length:
            raise ValueError(f"Vector slice length mismatch for key '{key}' (expected {length}, got {len(np_values)})")

        idx += length
        new_tensor = torch.tensor(np_values, dtype=value.dtype, device=value.device).view(value.shape)
        new_state_dict[key] = new_tensor

    if idx != len(vector):
        print(f"Warning: Unused vector elements: {len(vector) - idx}")

    return new_state_dict