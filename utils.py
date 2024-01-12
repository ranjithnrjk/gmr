import numpy as np
import pandas as pd
from typing import Union, List

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



def get_upper_triangular_values(matrix: np.ndarray) -> List[int]:
    """
    Extracts the values of the upper triangular matrix excluding diagonal elements.

    Parameters:
    - matrix (np.ndarray): The input matrix.

    Returns:
    - List[int]: List of values in the upper triangular matrix.
    """
    # Initialize an empty list to store upper triangular values
    upper_triangular_values = []

    # Iterate through each row
    for i in range(matrix.shape[0]):
        # Iterate through each column after the diagonal element
        for j in range(i + 1, matrix.shape[1]):
            # Append the value to the list
            upper_triangular_values.append(matrix[i, j])

    return upper_triangular_values

def threshold_transform(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Apply threshold transformation to a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - threshold (float): Threshold value. Default is 0.5.

    Returns:
    - pd.DataFrame: Transformed DataFrame with 1s and 0s.
    """
    # Apply the threshold transformation using a lambda function
    transformed_df = df.applymap(lambda x: 1 if x >= threshold else 0)

    return transformed_df


def group_by_category(dataframe: pd.DataFrame, category_column: str) -> pd.DataFrame:
    """
    Group rows based on a categorical column and display the grouped data.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - category_column (str): Name of the categorical column for grouping.

    Returns:
    - pd.DataFrame: Grouped DataFrame.
    """
    # Group the DataFrame by the specified category column
    grouped_df = dataframe.groupby(category_column)

    # Convert the grouped DataFrame to a new DataFrame
    grouped_result = grouped_df.apply(lambda x: x.reset_index(drop=True)).reset_index(drop=True)

    return grouped_result