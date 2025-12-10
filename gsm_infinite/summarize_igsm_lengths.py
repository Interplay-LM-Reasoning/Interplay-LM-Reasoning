#!/usr/bin/env python3
"""Summarize average problem lengths across all Igsm directory structures.

This script analyzes the entire Igsm folder structure including:
- zero_context/
- 8k/, 16k/, 32k/
Each with hard/medium difficulty levels and operation count subdirectories.

Usage:
    python gsm_infinite/summarize_igsm_lengths.py --root gsm_infinite/gsm-infinite/data/realistic/Igsm
"""

import argparse
import json
import os
import glob
import csv
from collections import defaultdict
import statistics


PREFERRED_KEYS = [
    "question",
    "problem", 
    "prompt",
    "text",
    "input",
    "query",
    "question_text",
    "body",
]


def extract_text(obj):
    """Extract text from JSON object, trying common field names."""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return str(obj)
    
    # Try preferred keys first
    for k in PREFERRED_KEYS:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k].strip()
    
    # Try any string value
    for v in obj.values():
        if isinstance(v, str) and v.strip():
            return v.strip()
    
    # Fallback: JSON dump
    return json.dumps(obj, ensure_ascii=False)


def analyze_file(filepath):
    """Analyze a single JSONL file and return length statistics."""
    char_lengths = []
    word_lengths = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    txt = extract_text(obj)
                except json.JSONDecodeError:
                    # Treat as plain text
                    txt = line
                except Exception as e:
                    print(f"Warning: Error processing line {line_num} in {filepath}: {e}")
                    continue
                
                char_lengths.append(len(txt))
                word_lengths.append(len(txt.split()))
    
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    if not char_lengths:
        return None
    
    return {
        'count': len(char_lengths),
        'avg_chars': statistics.mean(char_lengths),
        'median_chars': statistics.median(char_lengths),
        'std_chars': statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0,
        'min_chars': min(char_lengths),
        'max_chars': max(char_lengths),
        'avg_words': statistics.mean(word_lengths),
        'median_words': statistics.median(word_lengths),
        'std_words': statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0,
        'min_words': min(word_lengths),
        'max_words': max(word_lengths),
    }


def parse_path_components(filepath, root):
    """Parse filepath to extract context_type, difficulty, and op_count."""
    rel_path = os.path.relpath(filepath, root)
    parts = rel_path.split(os.sep)
    
    if len(parts) < 3:
        return None, None, None
    
    context_type = parts[0]  # zero_context, 8k, 16k, 32k
    difficulty = parts[1]    # hard, medium
    op_count = parts[2]      # 2, 3, 4, ..., 20
    
    return context_type, difficulty, op_count


def aggregate_stats(file_stats_list):
    """Aggregate statistics from multiple files."""
    if not file_stats_list:
        return None
    
    total_count = sum(stats['count'] for stats in file_stats_list)
    
    # Weight averages by count
    weighted_avg_chars = sum(stats['avg_chars'] * stats['count'] for stats in file_stats_list) / total_count
    weighted_avg_words = sum(stats['avg_words'] * stats['count'] for stats in file_stats_list) / total_count
    
    all_char_avgs = [stats['avg_chars'] for stats in file_stats_list]
    all_word_avgs = [stats['avg_words'] for stats in file_stats_list]
    
    return {
        'total_files': len(file_stats_list),
        'total_examples': total_count,
        'weighted_avg_chars': weighted_avg_chars,
        'weighted_avg_words': weighted_avg_words,
        'file_avg_chars_mean': statistics.mean(all_char_avgs),
        'file_avg_chars_std': statistics.stdev(all_char_avgs) if len(all_char_avgs) > 1 else 0,
        'file_avg_words_mean': statistics.mean(all_word_avgs),
        'file_avg_words_std': statistics.stdev(all_word_avgs) if len(all_word_avgs) > 1 else 0,
        'min_chars': min(stats['min_chars'] for stats in file_stats_list),
        'max_chars': max(stats['max_chars'] for stats in file_stats_list),
        'min_words': min(stats['min_words'] for stats in file_stats_list),
        'max_words': max(stats['max_words'] for stats in file_stats_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize Igsm problem lengths")
    parser.add_argument("--root", default="gsm_infinite/gsm-infinite/data/realistic/Igsm",
                       help="Root Igsm directory")
    parser.add_argument("--output", help="Optional CSV output file")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed breakdown by operation count")
    
    args = parser.parse_args()
    
    # Find all JSONL files
    pattern = os.path.join(args.root, "**/*.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        print(f"No JSONL files found under {args.root}")
        return
    
    print(f"Found {len(files)} JSONL files")
    
    # Organize by hierarchy
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for filepath in files:
        context_type, difficulty, op_count = parse_path_components(filepath, args.root)
        if context_type and difficulty and op_count:
            stats = analyze_file(filepath)
            if stats:
                hierarchy[context_type][difficulty][op_count].append(stats)
    
    # Generate summary
    print("\n" + "="*80)
    print("IGSM PROBLEM LENGTH SUMMARY")
    print("="*80)
    
    # Overall summary by context type and difficulty
    summary_rows = []
    
    for context_type in sorted(hierarchy.keys()):
        print(f"\n📁 {context_type.upper()}")
        print("-" * 60)
        
        for difficulty in sorted(hierarchy[context_type].keys()):
            # Aggregate all operation counts for this difficulty
            all_stats = []
            for op_count in hierarchy[context_type][difficulty]:
                all_stats.extend(hierarchy[context_type][difficulty][op_count])
            
            agg = aggregate_stats(all_stats)
            if agg:
                print(f"  {difficulty.ljust(8)}: "
                      f"{agg['total_examples']:>6} examples, "
                      f"{agg['weighted_avg_chars']:>6.1f} chars, "
                      f"{agg['weighted_avg_words']:>5.1f} words "
                      f"(across {agg['total_files']} files)")
                
                summary_rows.append({
                    'context_type': context_type,
                    'difficulty': difficulty,
                    'total_files': agg['total_files'],
                    'total_examples': agg['total_examples'],
                    'avg_chars': agg['weighted_avg_chars'],
                    'avg_words': agg['weighted_avg_words'],
                    'min_chars': agg['min_chars'],
                    'max_chars': agg['max_chars'],
                    'min_words': agg['min_words'],
                    'max_words': agg['max_words'],
                })
            
            # Detailed breakdown by operation count
            if args.detailed:
                print(f"    Operation count breakdown for {difficulty}:")
                for op_count in sorted(hierarchy[context_type][difficulty].keys(), key=int):
                    op_stats = aggregate_stats(hierarchy[context_type][difficulty][op_count])
                    if op_stats:
                        print(f"      op{op_count.rjust(2)}: "
                              f"{op_stats['total_examples']:>5} examples, "
                              f"{op_stats['weighted_avg_chars']:>6.1f} chars, "
                              f"{op_stats['weighted_avg_words']:>5.1f} words")
    
    # Cross-context comparison
    print(f"\n📊 CROSS-CONTEXT COMPARISON")
    print("-" * 60)
    
    context_totals = defaultdict(lambda: {'examples': 0, 'chars': 0, 'words': 0, 'files': 0})
    
    for row in summary_rows:
        ct = row['context_type']
        context_totals[ct]['examples'] += row['total_examples']
        context_totals[ct]['chars'] += row['avg_chars'] * row['total_examples']
        context_totals[ct]['words'] += row['avg_words'] * row['total_examples']
        context_totals[ct]['files'] += row['total_files']
    
    for context_type in sorted(context_totals.keys()):
        totals = context_totals[context_type]
        avg_chars = totals['chars'] / totals['examples'] if totals['examples'] else 0
        avg_words = totals['words'] / totals['examples'] if totals['examples'] else 0
        
        print(f"{context_type.ljust(15)}: "
              f"{totals['examples']:>7} examples, "
              f"{avg_chars:>6.1f} avg chars, "
              f"{avg_words:>5.1f} avg words "
              f"({totals['files']} files)")
    
    # Save CSV if requested
    if args.output:
        try:
            with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['context_type', 'difficulty', 'total_files', 'total_examples', 
                             'avg_chars', 'avg_words', 'min_chars', 'max_chars', 'min_words', 'max_words']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\n💾 Summary saved to {args.output}")
        except Exception as e:
            print(f"Error writing CSV: {e}")


if __name__ == "__main__":
    main()
