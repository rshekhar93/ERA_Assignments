from data.download_data import download_hindi_corpus
from tokenizer.bpe import HindiBPETokenizer
import os
import random
from tqdm import tqdm
from collections import Counter

def analyze_corpus(corpus: list) -> dict:
    """
    Analyze corpus statistics
    Args:
        corpus: List of sentences
    Returns:
        Dictionary of statistics
    """
    chars = set()
    words = set()
    char_freq = Counter()
    word_freq = Counter()
    total_chars = 0
    total_words = 0
    
    for text in corpus:
        # Character statistics
        chars.update(text)
        char_freq.update(text)
        total_chars += len(text)
        
        # Word statistics
        text_words = text.split()
        words.update(text_words)
        word_freq.update(text_words)
        total_words += len(text_words)
    
    return {
        'unique_chars': len(chars),
        'unique_words': len(words),
        'total_chars': total_chars,
        'total_words': total_words,
        'avg_word_len': total_chars / total_words if total_words > 0 else 0,
        'top_chars': char_freq.most_common(10),
        'top_words': word_freq.most_common(10)
    }

def train_and_evaluate_tokenizer(num_sentences=100000, target_compression=3.3):
    """
    Train the tokenizer and evaluate its performance
    Args:
        num_sentences: Number of sentences to use for training
        target_compression: Target compression ratio (must be >= 3.3)
    """
    # Validate parameters
    if target_compression < 3.3:
        print("Warning: Target compression ratio must be at least 3.3. Setting to 3.3.")
        target_compression = 3.3
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Download corpus if not already downloaded
    corpus_path = 'data/hindi_corpus.txt'
    if not os.path.exists(corpus_path):
        print("Downloading corpus...")
        corpus = download_hindi_corpus(num_sentences)
    else:
        print("Loading existing corpus...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.read().splitlines()
    
    # Analyze corpus
    print("\nAnalyzing corpus...")
    stats = analyze_corpus(corpus)
    
    print("\nInitial Corpus Statistics:")
    print("-" * 80)
    print(f"Total sentences: {len(corpus):,}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Unique characters: {stats['unique_chars']:,}")
    print(f"Unique words: {stats['unique_words']:,}")
    print(f"Average word length: {stats['avg_word_len']:.2f} characters")
    
    print("\nMost common characters:")
    for char, count in stats['top_chars']:
        print(f"'{char}': {count:,} times")
    
    print("\nMost common words:")
    for word, count in stats['top_words']:
        print(f"'{word}': {count:,} times")
    print("-" * 80)
    
    # Initialize and train tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = HindiBPETokenizer(target_compression=target_compression, max_vocab_size=5000)
    
    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train(corpus)
    
    # Verify requirements are met
    vocab_size = len(tokenizer.token_to_id)
    if vocab_size > 5000:
        print(f"\nError: Vocabulary size ({vocab_size}) exceeds maximum limit of 5000 tokens!")
        return
    
    # Save the trained tokenizer
    model_path = 'models/hindi_tokenizer.json'
    tokenizer.save(model_path)
    print(f"\nSaved trained tokenizer to {model_path}")
    
    # Evaluate on some example sentences
    print("\nEvaluating tokenizer on example sentences...")
    
    # Select some random sentences from corpus for testing
    test_sentences = random.sample(corpus, min(5, len(corpus)))
    
    # Add some custom test sentences
    test_sentences.extend([
        "नमस्ते दुनिया",  # Hello World
        "मैं हिंदी सीख रहा हूं",  # I am learning Hindi
        "भारत एक विशाल देश है",  # India is a vast country
    ])
    
    total_compression = 0
    
    print("\nExample tokenizations:")
    print("-" * 80)
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)
        decoded = tokenizer.decode(tokens)
        compression = len(sentence) / len(tokens)
        total_compression += compression
        
        print(f"Original    : {sentence}")
        print(f"Tokens      : {tokens}")
        print(f"Token Count : {len(tokens)}")
        print(f"Decoded     : {decoded}")
        print(f"Compression : {compression:.2f}")
        print("-" * 80)
    
    avg_compression = total_compression / len(test_sentences)
    print(f"\nFinal Results:")
    print(f"Vocabulary size: {len(tokenizer.token_to_id)} tokens")
    print(f"Average compression ratio: {avg_compression:.2f}")
    
    # Final verification
    if avg_compression < 3.3:
        print("\nWarning: Final compression ratio is below target of 3.3!")

if __name__ == "__main__":
    train_and_evaluate_tokenizer(
        num_sentences=100000,    # Number of sentences to use for training
        target_compression=3.3   # Target compression ratio
    ) 