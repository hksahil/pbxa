from src.constants import POWERBI_TYPE_CODES
import pandas as pd


def clean_text(text):
    """
    Remove invisible Unicode characters like Left-to-Right Mark, Right-to-Left Mark, etc.

    Args:
        text: String to clean

    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        return text

    # Remove common invisible Unicode characters
    invisible_chars = [
        "\u200e",  # Left-to-Right Mark (LRM)
        "\u200f",  # Right-to-Left Mark (RLM)
        "\u202a",  # Left-to-Right Embedding
        "\u202b",  # Right-to-Left Embedding
        "\u202c",  # Pop Directional Formatting
        "\u202d",  # Left-to-Right Override
        "\u202e",  # Right-to-Left Override
        "\ufeff",  # Zero Width No-Break Space (BOM)
        "\u200b",  # Zero Width Space
        "\u200c",  # Zero Width Non-Joiner
        "\u200d",  # Zero Width Joiner
    ]

    cleaned = text
    for char in invisible_chars:
        cleaned = cleaned.replace(char, "")

    return cleaned


def get_type_name(type_code):
    """
    Convert Power BI type code to readable type name.

    Args:
        type_code: Numeric type code from Power BI

    Returns:
        String representation of the type
    """
    return POWERBI_TYPE_CODES.get(type_code, f"Type Code {type_code}")


def filter_dataframe_by_text(df, search_text):
    """
    Filter a DataFrame by searching for text across all columns.

    Args:
        df: pandas DataFrame to filter
        search_text: Text to search for (case-insensitive)

    Returns:
        Filtered DataFrame containing only rows that match the search text
    """
    if df is None or df.empty or not search_text or search_text.strip() == "":
        return df
    
    search_text = search_text.strip().lower()
    
    # Create a boolean mask for rows containing the search text
    mask = pd.Series([False] * len(df), index=df.index)
    
    # Search across all columns
    for col in df.columns:
        try:
            # Convert column to string and search (case-insensitive)
            mask |= df[col].astype(str).str.lower().str.contains(search_text, na=False, regex=False)
        except Exception:
            # Skip columns that can't be converted or searched
            continue
    
    return df[mask]

