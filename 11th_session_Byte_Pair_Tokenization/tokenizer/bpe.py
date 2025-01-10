import json
import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple
from .utils import get_char_pairs, merge_pairs, calculate_compression_ratio
from tqdm import tqdm
import random

class HindiBPETokenizer:
    def __init__(self, target_compression: float = 3.3, max_vocab_size: int = 5000):
        """
        Initialize Hindi BPE Tokenizer
        Args:
            target_compression: Target compression ratio (default: 3.3)
            max_vocab_size: Maximum vocabulary size (default: 5000)
        """
        self.target_compression = target_compression
        self.max_vocab_size = max_vocab_size
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
    
    def _calculate_compression(self, sample_texts: List[str]) -> float:
        """Calculate compression ratio on sample texts"""
        total_chars = sum(len(text) for text in sample_texts)
        total_tokens = sum(len(self.encode(text)) for text in sample_texts)
        return total_chars / total_tokens if total_tokens > 0 else 0
    
    def train_from_frequencies(self, word_freqs: Dict[str, int], sample_texts: List[str]):
        """Train BPE tokenizer using pre-computed word frequencies"""
        # Initialize with characters
        chars = set()
        for word in word_freqs.keys():
            chars.update(word.split())
        
        # Add frequent characters to vocabulary
        char_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for char in word.split():
                char_freqs[char] += freq
        
        # Keep top 200 characters
        top_chars = sorted(char_freqs.items(), key=lambda x: x[1], reverse=True)[:200]
        for char, _ in top_chars:
            if char not in self.token_to_id:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = char
        
        print(f"\nInitial Statistics:")
        print(f"Characters in vocabulary: {len(self.token_to_id)}")
        
        # BPE training loop
        pbar = tqdm(total=self.max_vocab_size, desc="Training BPE")
        
        # Sample texts for compression calculation
        eval_texts = random.sample(sample_texts, min(1000, len(sample_texts)))
        
        while len(self.token_to_id) < self.max_vocab_size:
            # Get character pairs and their frequencies
            pairs = get_char_pairs(word_freqs)
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Add to vocabulary
            self.merges[best_pair] = len(self.token_to_id)
            self.token_to_id[best_pair[0] + best_pair[1]] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = best_pair[0] + best_pair[1]
            
            # Update word frequencies
            word_freqs = merge_pairs(word_freqs, best_pair)
            
            # Check compression ratio every 100 merges
            if len(self.token_to_id) % 100 == 0:
                compression = self._calculate_compression(eval_texts)
                pbar.set_postfix({'compression': f'{compression:.2f}'})
                
                if compression >= self.target_compression:
                    print(f"\nReached target compression ratio: {compression:.2f}")
                    break
            
            pbar.update(1)
        
        pbar.close()
        
        # Final statistics
        compression = self._calculate_compression(eval_texts)
        print(f"\nFinal Statistics:")
        print(f"Vocabulary size: {len(self.token_to_id)}")
        print(f"Compression ratio: {compression:.2f}")
    
    def train(self, corpus: List[str]):
        """Train BPE tokenizer on Hindi corpus"""
        print("Computing word frequencies...")
        word_freqs = defaultdict(int)
        for text in tqdm(corpus, desc="Processing sentences"):
            words = text.split()
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
        
        self.train_from_frequencies(word_freqs, corpus)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode Hindi text to token IDs
        Args:
            text: Input Hindi text
        Returns:
            List of token IDs
        """
        tokens = []
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            while True:
                pairs = get_char_pairs({word: 1})
                if not pairs:
                    break
                
                bigram = min(pairs.keys(), key=lambda x: self.merges.get(x, float('inf')))
                if bigram not in self.merges:
                    break
                
                word = merge_pairs({word: 1}, bigram)
            
            word_tokens = word.split()
            for token in word_tokens:
                tokens.append(self.token_to_id.get(token, self.token_to_id['<UNK>']))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to Hindi text
        Args:
            tokens: List of token IDs
        Returns:
            Decoded Hindi text
        """
        # Convert tokens to text
        words = []
        current_word = []
        
        for token in tokens:
            if token in self.id_to_token:
                token_text = self.id_to_token[token]
                
                # Skip special tokens
                if token_text in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    continue
                    
                # If token contains end of word marker
                if '</w>' in token_text:
                    # Remove </w> and add to current word
                    token_text = token_text.replace('</w>', '')
                    # Skip if token is not Hindi (contains ASCII characters)
                    if not any(ord(c) > 127 for c in token_text) and token_text not in ['।', '॥']:
                        continue
                    current_word.append(token_text)
                    # Join subwords and add to words list
                    words.append(''.join(current_word))
                    current_word = []
                else:
                    # Skip if token is not Hindi (contains ASCII characters)
                    if not any(ord(c) > 127 for c in token_text) and token_text not in ['।', '॥']:
                        continue
                    current_word.append(token_text)
        
        # Handle any remaining subwords
        if current_word:
            words.append(''.join(current_word))
        
        # Join words with spaces
        return ' '.join(words)
    
    def get_compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio for given text
        Args:
            text: Input Hindi text
        Returns:
            Compression ratio
        """
        encoded = self.encode(text)
        return calculate_compression_ratio(text, encoded)
    
    def save(self, path: str):
        """
        Save tokenizer files
        Args:
            path: Path to save tokenizer
        """
        data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'merges': {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """
        Load tokenizer files
        Args:
            path: Path to load tokenizer from
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        self.merges = {tuple(k.split()): v for k, v in data['merges'].items()} 