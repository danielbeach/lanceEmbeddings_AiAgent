# Lance Embeddings AI Agent

A RAG (Retrieval Augmented Generation) system for querying blog posts using LanceDB vector database and LangChain.

## Overview

This project processes Substack blog post exports (CSV metadata + HTML files), generates embeddings, stores them in a Lance dataset, and provides an interactive chat interface to query your blog content using AI.

## Features

- Extract and process blog posts from CSV and HTML files
- Generate embeddings using sentence-transformers
- Store data in Lance format for efficient vector search
- Interactive chat interface powered by OpenAI GPT
- Query your blog posts using natural language

## Setup

### Prerequisites

- Python 3.12+
- OpenAI API key (for chat functionality)

### Installation

1. Install dependencies:
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

2. Set your OpenAI API key (optional, will prompt if not set):
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### 1. Extract Post Data and Generate Embeddings

Process your blog posts and create the Lance dataset:

```bash
uv run extract_post_data.py
```

This script:
- Reads `posts_summary.csv` for post metadata
- Extracts text from HTML files in the `posts/` directory
- Filters to published posts (`is_published = true`)
- Generates embeddings using `all-MiniLM-L6-v2` model
- Creates `posts.lance` dataset with embeddings

**Output**: `posts.lance` - A Lance dataset containing:
- `post_id`: Unique post identifier
- `title`: Post title
- `sub_title`: Post subtitle
- `post_date`: Publication date
- `blog_text`: Full post content
- `embedding`: 384-dimensional vector embedding

### 2. Inspect the Lance Dataset

View sample rows from your dataset:

```bash
uv run inspect_lance.py
```

Displays:
- First 10 rows with key fields
- Embedding dimensions
- Total row count

### 3. Chat with Your Blog Posts

Start an interactive chat session to query your blog content:

```bash
uv run chat.py
```

Example queries:
- "Have I written anything about Excel in Databricks?"
- "What are my thoughts on vector databases?"
- "Tell me about my posts on data engineering"

The chat interface:
- Retrieves the top 5 most relevant posts based on your question
- Uses GPT-4o-mini to generate answers based on the retrieved context
- Shows source citations (post titles and IDs)

Type `quit` or `exit` to end the chat session.

## Project Structure

```
.
├── extract_post_data.py    # Main script to process posts and create Lance dataset
├── chat.py                  # Interactive chat interface
├── inspect_lance.py         # Dataset inspection tool
├── posts_summary.csv        # Input: Post metadata CSV
├── posts/                   # Input: Directory with HTML files
├── posts.lance              # Output: Lance dataset with embeddings
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Dependencies

- `duckdb` - Database operations
- `beautifulsoup4` - HTML parsing
- `lxml` - HTML parser backend
- `pyarrow` - Data format support
- `pytz` - Timezone handling
- `sentence-transformers` - Embedding generation
- `langchain` - RAG framework
- `langchain-community` - Community integrations
- `langchain-openai` - OpenAI integration
- `langchain-huggingface` - HuggingFace embeddings
- `lancedb` - Vector database

## How It Works

1. **Data Extraction** (`extract_post_data.py`):
   - Reads post metadata from CSV
   - Matches posts to HTML files by numeric ID
   - Extracts and cleans text from HTML
   - Combines title, subtitle, and content for embedding
   - Generates embeddings using sentence-transformers
   - Writes to Lance format

2. **Chat Interface** (`chat.py`):
   - Loads the Lance dataset
   - Embeds user questions using the same model
   - Performs vector similarity search
   - Retrieves top 5 relevant posts
   - Sends context to GPT-4o-mini for answer generation
   - Displays answer with source citations

## Configuration

### Input Files

- `posts_summary.csv`: Must contain columns:
  - `post_id`: Post identifier (e.g., "184371353.from-dba-to-data-everything")
  - `is_published`: Boolean or string "true"/"false"
  - `title`, `subtitle`/`sub_title`, `post_date`: Optional metadata

- `posts/`: Directory containing HTML files with numeric IDs in filenames

### Output

- `posts.lance`: Lance dataset ready for vector search

## Performance

The script includes timing information for:
- Embedding generation time
- Lance dataset write time

Typical performance:
- Embedding generation: ~1-2 seconds per 100 posts
- Lance write: <1 second for 290 posts

## Notes

- The embedding model (`all-MiniLM-L6-v2`) generates 384-dimensional vectors
- Only posts with `is_published = true` are processed
- HTML files are matched by numeric ID extracted from filenames
- The first matching HTML file is used if multiple matches exist
