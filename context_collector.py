import os
import argparse
from typing import List, Dict
import json


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


def format_for_chatgpt(contents: Dict[str, str]) -> str:
    """Formats the collected contents for ChatGPT input.

    Args:
        contents: Dict of file paths and their contents

    Returns:
        Formatted string ready for ChatGPT
    """
    output = []
    
    # Sort files by name for consistent output
    for file_path in sorted(contents.keys()):
        content = contents[file_path]
        # Add file separator for clarity
        output.append(f"\n{'='*80}\nFile: {file_path}\n{'='*80}\n```\n{content}\n```\n")
        
    return "\n".join(output)


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
    formatted_output = format_for_chatgpt(contents)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(formatted_output)
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main() 