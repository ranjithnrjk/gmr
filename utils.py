import numpy as np
from typing import Union

def cosine_sim(
        vector1: Union[list, np.array], 
        vector2: Union[list, np.array]
        ) -> Union[float, int]:
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vector1: A NumPy array representing the first vector.
        vector2: A NumPy array representing the second vector.

    Returns:
        The cosine similarity between the two vectors.
    """
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)
