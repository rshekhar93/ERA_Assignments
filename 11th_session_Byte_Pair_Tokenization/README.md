# Hindi BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer specifically designed for the Hindi language. This tokenizer achieves efficient text compression while maintaining a vocabulary size of less than 5000 tokens and a compression ratio of 3.2 or higher.

## Features

- Custom BPE implementation for Hindi text
- Vocabulary size < 5000 tokens
- Compression ratio > 3.2
- Efficient encoding and decoding
- Support for special tokens (PAD, UNK, BOS, EOS)
- Web interface using Gradio
- Comprehensive training statistics

## Project Structure

```
.
├── data/
│   ├── download_data.py    # Script to download Hindi corpus
│   └── hindi_corpus.txt    # Downloaded corpus (generated)
├── tokenizer/
│   ├── __init__.py        # Package initialization
│   ├── bpe.py             # Main BPE implementation
│   └── utils.py           # Utility functions
├── app.py                 # Gradio web interface
├── train_tokenizer.py     # Training script
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hindi-bpe-tokenizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Tokenizer

1. Download the Hindi corpus:
```bash
python data/download_data.py
```

2. Train the tokenizer:
```bash
python train_tokenizer.py
```

### Using the Tokenizer

```python
from tokenizer import HindiBPETokenizer

# Initialize tokenizer
tokenizer = HindiBPETokenizer()

# Load trained tokenizer
tokenizer.load('tokenizer.json')

# Encode text
text = "मैं हिंदी में लिख रहा हूं"
tokens = tokenizer.encode(text)

# Decode tokens
decoded_text = tokenizer.decode(tokens)
```

### Web Interface

Run the Gradio web interface:
```bash
python app.py
```

Visit `http://localhost:7860` in your browser to access the interface.

## Training Details

- **Corpus**: Hindi text corpus with quality filters
- **Vocabulary Size**: < 5000 tokens
- **Compression Ratio**: > 3.2
- **Special Tokens**:
  - `<PAD>`: Padding token (ID: 0)
  - `<UNK>`: Unknown token (ID: 1)
  - `<BOS>`: Beginning of sequence (ID: 2)
  - `<EOS>`: End of sequence (ID: 3)

## Example

```python
# Example usage
text = "नमस्ते दुनिया"
tokens = tokenizer.encode(text)
print(f"Original text: {text}")
print(f"Encoded tokens: {tokens}")
print(f"Decoded text: {tokenizer.decode(tokens)}")
print(f"Compression ratio: {tokenizer.get_compression_ratio(text):.2f}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.