import regex as re
from collections import defaultdict
from typing import Dict, Tuple, List, Union

def get_char_pairs(word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """
    Get all character pairs and their frequencies from words
    Args:
        word_freqs: Dictionary of word frequencies
    Returns:
        Dictionary of character pairs and their frequencies
    """
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        chars = word.split()
        for i in range(len(chars) - 1):
            pairs[(chars[i], chars[i + 1])] += freq
    return pairs

def merge_pairs(word_freqs: Dict[str, int], pair: Tuple[str, str]) -> Union[Dict[str, int], str]:
    """
    Merge all occurrences of a character pair in all words
    Args:
        word_freqs: Dictionary of word frequencies
        pair: Tuple of characters to merge
    Returns:
        If word_freqs has one item, returns merged word as string
        Otherwise, returns dictionary of updated word frequencies
    """
    # Create the target and replacement strings
    target = f"{pair[0]} {pair[1]}"
    replacement = f"{pair[0]}{pair[1]}"
    
    # If we're processing a single word (encoding)
    if len(word_freqs) == 1 and 1 in word_freqs.values():
        word = next(iter(word_freqs.keys()))
        parts = word.split()
        merged_parts = []
        i = 0
        while i < len(parts) - 1:
            if parts[i] == pair[0] and parts[i + 1] == pair[1]:
                merged_parts.append(replacement)
                i += 2
            else:
                merged_parts.append(parts[i])
                i += 1
        if i < len(parts):
            merged_parts.append(parts[-1])
        return " ".join(merged_parts)
    
    # If we're processing multiple words (training)
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        parts = word.split()
        merged_parts = []
        i = 0
        while i < len(parts) - 1:
            if parts[i] == pair[0] and parts[i + 1] == pair[1]:
                merged_parts.append(replacement)
                i += 2
            else:
                merged_parts.append(parts[i])
                i += 1
        if i < len(parts):
            merged_parts.append(parts[-1])
        new_word = " ".join(merged_parts)
        new_word_freqs[new_word] = freq
    
    return new_word_freqs

def calculate_compression_ratio(text: str, encoded_tokens: List[int]) -> float:
    """
    Calculate compression ratio between original text and encoded tokens
    Args:
        text: Original input text
        encoded_tokens: List of encoded token IDs
    Returns:
        Compression ratio (characters/tokens)
    """
    return len(text) / len(encoded_tokens) 