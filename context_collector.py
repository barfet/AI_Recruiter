import os
import argparse
from typing import List, Dict, Tuple
import json
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model to use for counting (affects tokenizer)
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def collect_file_contents(folder_paths: List[str], ignore_patterns: List[str] = None) -> Dict[str, str]:
    """Recursively collects contents of all files in the given folders.

    Args:
        folder_paths: List of paths to process
        ignore_patterns: List of patterns to ignore (e.g. ["*.pyc", "__pycache__"])

    Returns:
        Dict mapping relative file paths to their contents
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".env",
            "venv",
            "node_modules",
            ".DS_Store",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "*.so"
        ]

    result = {}
    
    for folder_path in folder_paths:
        for root, dirs, files in os.walk(folder_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ignore_patterns)]
            
            for file in files:
                # Skip ignored files
                if any(pattern in file for pattern in ignore_patterns):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    # Get relative path from the root folder
                    rel_path = os.path.relpath(file_path, folder_path)
                    
                    # Try to read the file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            result[f"{os.path.basename(folder_path)}/{rel_path}"] = content
                    except UnicodeDecodeError:
                        print(f"Skipping binary file: {rel_path}")
                        continue
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
                    
    return result


def format_for_chatgpt(contents: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    """Formats the collected contents for ChatGPT input and counts tokens.

    Args:
        contents: Dict of file paths and their contents

    Returns:
        Tuple of (formatted string, dict with token counts for different models)
    """
    output = []
    total_chars = 0
    
    # Sort files by name for consistent output
    for file_path in sorted(contents.keys()):
        content = contents[file_path]
        # Add file separator for clarity
        formatted = f"\n{'='*80}\nFile: {file_path}\n{'='*80}\n```\n{content}\n```\n"
        output.append(formatted)
        total_chars += len(formatted)
        
    final_output = "\n".join(output)
    
    # Count tokens for different models
    token_counts = {
        "gpt-4": count_tokens(final_output, "gpt-4"),
        "gpt-3.5-turbo": count_tokens(final_output, "gpt-3.5-turbo"),
        "claude-2": count_tokens(final_output, "gpt-4"),  # Claude uses similar tokenization
    }
    
    return final_output, token_counts


def print_token_info(token_counts: Dict[str, int]) -> None:
    """Print token count information and model compatibility."""
    print("\nToken count information:")
    print("-" * 40)
    
    # Model context limits
    model_limits = {
        "gpt-4": 8192,
        "gpt-3.5-turbo": 4096,
        "claude-2": 100000
    }
    
    for model, count in token_counts.items():
        limit = model_limits.get(model, 0)
        print(f"{model}:")
        print(f"  Tokens: {count:,}")
        print(f"  Context limit: {limit:,}")
        print(f"  Status: {'✅ Fits' if count <= limit else '❌ Too large'}")
        print(f"  Usage: {count/limit*100:.1f}% of context window")
        print()


def main():
    parser = argparse.ArgumentParser(description="Collect and format file contents for ChatGPT")
    parser.add_argument("folders", nargs="+", help="One or more folder paths to process")
    parser.add_argument("--output", "-o", required=True, help="Output file path (required)")
    parser.add_argument("--ignore", "-i", nargs="+", help="Additional patterns to ignore")
    
    args = parser.parse_args()
    
    ignore_patterns = None
    if args.ignore:
        ignore_patterns = args.ignore
        
    contents = collect_file_contents(args.folders, ignore_patterns)
    formatted_output, token_counts = format_for_chatgpt(contents)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(formatted_output)
    print(f"\nOutput written to {args.output}")
    
    print_token_info(token_counts)


if __name__ == "__main__":
    main() 