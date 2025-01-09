import json
import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple
from .utils import get_char_pairs, merge_pairs, calculate_compression_ratio
from tqdm import tqdm
import random

class HindiBPETokenizer:
    def __init__(self, target_compression: float = 3.0, max_vocab_size: int = 50000):
        """
        Initialize Hindi BPE Tokenizer
        Args:
            target_compression: Target compression ratio (default: 3.0)
            max_vocab_size: Maximum vocabulary size (default: 50000)
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
    
    def _calculate_avg_compression(self, sample_texts: List[str], num_samples: int = 100) -> float:
        """
        Calculate average compression ratio on sample texts
        Args:
            sample_texts: List of texts to sample from
            num_samples: Number of samples to use
        Returns:
            Average compression ratio
        """
        if len(sample_texts) > num_samples:
            texts = random.sample(sample_texts, num_samples)
        else:
            texts = sample_texts
        
        total_compression = 0
        for text in texts:
            tokens = self.encode(text)
            compression = len(text) / len(tokens)
            total_compression += compression
        
        return total_compression / len(texts)
    
    def train_from_frequencies(self, word_freqs: Dict[str, int], sample_texts: List[str]):
        """
        Train BPE tokenizer using pre-computed word frequencies
        Args:
            word_freqs: Dictionary of word frequencies
            sample_texts: List of texts to calculate compression ratio
        """
        # Initialize vocabulary with characters
        chars = set()
        for word in word_freqs.keys():
            chars.update(word.split())
        
        # Add characters to vocabulary
        for char in sorted(chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = char
        
        # Print initial statistics
        print(f"\nInitial statistics:")
        print(f"Number of unique characters (initial tokens): {len(chars)}")
        print(f"Number of unique words before BPE: {len(word_freqs)}")
        print(f"Total vocabulary size (including special tokens): {len(self.token_to_id)}\n")
        
        # BPE training loop
        pbar = tqdm(total=self.max_vocab_size, desc="Training BPE")
        current_vocab_size = len(self.token_to_id)
        
        while current_vocab_size < self.max_vocab_size:
            # Check compression ratio every 100 merges
            if current_vocab_size % 100 == 0:
                avg_compression = self._calculate_avg_compression(sample_texts)
                pbar.set_postfix({'compression': f'{avg_compression:.2f}'})
                
                if avg_compression >= self.target_compression:
                    print(f"\nReached target compression ratio: {avg_compression:.2f}")
                    break
            
            pairs = get_char_pairs(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = len(self.token_to_id)
            self.token_to_id[best_pair[0] + best_pair[1]] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = best_pair[0] + best_pair[1]
            
            # Update word frequencies with merged pairs
            word_freqs = merge_pairs(word_freqs, best_pair)
            current_vocab_size = len(self.token_to_id)
            pbar.update(1)
        
        pbar.close()
        print(f"Final vocabulary size: {len(self.token_to_id)}")
    
    def train(self, corpus: List[str]):
        """
        Train BPE tokenizer on Hindi corpus
        Args:
            corpus: List of Hindi sentences
        """
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
        text = []
        for token in tokens:
            if token in self.id_to_token:
                text.append(self.id_to_token[token])
        
        # Remove special tokens and join
        text = ' '.join(text)
        text = text.replace(' </w>', '')
        text = re.sub(r'\s+', '', text)
        return text
    
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