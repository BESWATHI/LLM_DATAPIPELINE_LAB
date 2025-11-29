# LLM Data Pipeline Lab

This lab demonstrates how to build an efficient data pipeline for training Large Language Models (LLMs) using PyTorch, Hugging Face Transformers, and the Datasets library.

## Overview

The lab walks through the process of:
1. Loading a text dataset (WikiText-2)
2. Setting up a tokenizer from a pre-trained model
3. Efficiently tokenizing text data in batches
4. Creating tokenized blocks for language modeling
5. Building a PyTorch Dataset and DataLoader for training

## Features

- **Batch Tokenization**: Efficiently tokenizes large datasets in batches to optimize memory usage
- **Block-based Processing**: Splits tokenized data into fixed-size blocks suitable for causal language modeling
- **PyTorch Integration**: Creates a custom Dataset class and DataLoader for seamless integration with PyTorch training loops
- **Hugging Face Integration**: Uses Hugging Face's Transformers library for tokenization and model access

## Dataset

This lab uses the **WikiText-2** dataset, which contains raw Wikipedia text suitable for language modeling tasks. The dataset includes approximately 36,718 lines of text.

## Model

The lab uses the **facebook/opt-125m** tokenizer, which provides:
- Vocabulary size: 50,265 tokens
- Fast tokenization support
- Proper handling of padding tokens

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)

See `requirements.txt` for specific package versions.

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook LLM_Datapipeline_Lab.ipynb
```

2. Run the cells sequentially to:
   - Load the WikiText-2 dataset
   - Initialize the tokenizer
   - Tokenize the dataset in batches
   - Create token blocks for training
   - Set up the PyTorch Dataset and DataLoader

## Key Components

### 1. Dataset Loading
- Uses Hugging Face's `datasets` library to load WikiText-2
- Accesses the raw text content for tokenization

### 2. Tokenization
- Batch tokenization for efficiency
- Configurable batch size (default: 512)
- Adds EOS tokens between text segments

### 3. Block Creation
- Splits tokenized data into fixed-size blocks (default: 128 tokens)
- Ensures all training examples have consistent length
- Creates non-overlapping blocks for training

### 4. PyTorch Dataset & DataLoader
- Custom `BlockDataset` class that returns token blocks
- Labels are identical to input_ids (for causal LM training)
- DataLoader with configurable batch size (default: 8)
- Includes shuffling for better training

## Configuration Parameters

- `block_size`: 128 (number of tokens per block)
- `batch_tokenize_batch_size`: 512 (batch size for tokenization)
- `train_batch_size`: 8 (batch size for training)

You can modify these parameters in the notebook to experiment with different configurations.

## Output

After running the pipeline, you'll have:
- A tokenized dataset with approximately 18,973 blocks of 128 tokens each
- A PyTorch DataLoader ready for training (approximately 2,372 batches per epoch)
- Batch examples showing the shape and decoded text content

## Notes

- The tokenizer automatically handles padding tokens (uses EOS token as pad token if needed)
- All sequences are fixed-length after block creation, simplifying batching
- The pipeline is designed to handle large datasets efficiently through batch processing

## Future Enhancements

Potential improvements for this pipeline:
- Add data validation and cleaning steps
- Implement dynamic padding for variable-length sequences
- Add data augmentation techniques
- Support for multiple datasets
- Add progress bars for long-running operations
- Implement caching for tokenized data



