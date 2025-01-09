import gradio as gr
from tokenizer.bpe import HindiBPETokenizer
from data.download_data import download_hindi_corpus

def initialize_tokenizer():
    """Initialize and train tokenizer if not already trained"""
    try:
        tokenizer = HindiBPETokenizer()
        tokenizer.load('tokenizer/hindi_tokenizer.json')
        print("Loaded pre-trained tokenizer")
    except:
        print("Training new tokenizer...")
        corpus = download_hindi_corpus(num_sentences=1000000)
        tokenizer = HindiBPETokenizer(vocab_size=4800)
        tokenizer.train(corpus)
        tokenizer.save('tokenizer/hindi_tokenizer.json')
    return tokenizer

def process_text(text: str, mode: str) -> str:
    """Process text based on mode (encode/decode)"""
    if not text.strip():
        return "Please enter some text"
    
    if mode == "Encode":
        tokens = tokenizer.encode(text)
        ratio = tokenizer.get_compression_ratio(text)
        return (f"Token IDs: {tokens}\n"
                f"Number of tokens: {len(tokens)}\n"
                f"Compression ratio: {ratio:.2f}\n"
                f"Vocabulary size: {len(tokenizer.token_to_id)}")
    else:
        try:
            tokens = [int(t) for t in text.split()]
            decoded = tokenizer.decode(tokens)
            return f"Decoded Text: {decoded}"
        except:
            return "Error: Please input valid token IDs (space-separated integers)"

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = initialize_tokenizer()

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Enter Hindi text to encode or token IDs to decode"
        ),
        gr.Radio(
            ["Encode", "Decode"],
            label="Mode",
            value="Encode"
        )
    ],
    outputs=gr.Textbox(label="Output"),
    title="Hindi BPE Tokenizer",
    description="""
    A Byte Pair Encoding (BPE) based tokenizer for Hindi text.
    
    Features:
    - Vocabulary size: <5000 tokens
    - Compression ratio: >3.2
    - Trained on OSCAR Hindi corpus
    
    Examples:
    1. Encode: Enter Hindi text like "भारत एक विशाल देश है"
    2. Decode: Enter space-separated token IDs like "1958 1783 3939 1760 1977 1815"
    """,
    examples=[
        ["नमस्ते दुनिया", "Encode"],
        ["भारत एक विशाल देश है", "Encode"],
        ["1958 1783 3939 1760 1977 1815", "Decode"]
    ]
)

if __name__ == "__main__":
    demo.launch() 