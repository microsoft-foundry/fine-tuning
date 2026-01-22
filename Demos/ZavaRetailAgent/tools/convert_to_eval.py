"""
Convert to Eval Dataset Script

This script converts SFT (Supervised Fine-Tuning) training data to eval format
and then expands it to create intermediate tool call training samples.

Two-stage process:
1. Convert SFT format to Eval format by extracting the last assistant tool_calls as expected_output
2. Expand eval items to create intermediate training samples (one per tool call)

Usage:
    python convert_to_eval.py
    
Edit the INPUT_FILE variable below to test different files.
"""

import json
import copy
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# ===== EDIT THIS LINE TO TRY DIFFERENT FILES =====
INPUT_FILE = "data/sft_test.jsonl"
# =================================================

def convert_sft_to_eval(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert an SFT item to eval format by extracting the last assistant tool_calls as expected_output.
    
    Args:
        item: SFT item with {"messages": [...], "tools": [...]}
        
    Returns:
        Eval format item with {"item": {"messages": [...], "tools": [...], "expected_output": {...}}}
        or None if no tool calls found
    """
    messages = item.get("messages", [])
    tools = item.get("tools", [])
    
    if not messages:
        return None
    
    # Find the last assistant message with tool_calls
    last_tool_call_message = None
    last_tool_call_index = -1
    
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and "tool_calls" in messages[i]:
            last_tool_call_message = messages[i]
            last_tool_call_index = i
            break
    
    if not last_tool_call_message:
        return None  # Skip conversations without tool calls
    
    # Create expected_output from the last assistant tool_calls
    expected_output = {
        "role": "assistant",
        "tool_calls": last_tool_call_message["tool_calls"]
    }
    
    # Create messages list - keep everything up to but NOT including the last assistant tool_call message
    # Remove all messages after and including the last tool call
    eval_messages = copy.deepcopy(messages[:last_tool_call_index])
    
    # Wrap in eval format with "item" wrapper
    eval_item = {
        "item": {
            "messages": eval_messages,
            "tools": tools,
            "expected_output": expected_output
        }
    }
    
    return eval_item

def extract_tool_calls_from_messages(messages: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Extract tool calls from messages and return list of (message_index, tool_call) tuples.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of tuples containing (message_index, tool_call_dict)
    """
    tool_calls = []
    
    for i, message in enumerate(messages):
        if message.get("role") == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                tool_calls.append((i, tool_call))
    
    return tool_calls

def create_truncated_conversation(messages: List[Dict[str, Any]], up_to_index: int, target_tool_call: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a truncated conversation up to a specific message index with the target tool call as expected output.
    
    Args:
        messages: Original messages list
        up_to_index: Index to truncate messages at (inclusive)
        target_tool_call: The tool call to use as expected output
        
    Returns:
        Tuple of (truncated_messages, expected_output)
    """
    # Include all messages up to and including the target message index
    truncated_messages = copy.deepcopy(messages[:up_to_index + 1])
    
    # Remove tool_calls from the last assistant message (we'll put it in expected_output)
    if truncated_messages and truncated_messages[-1].get("role") == "assistant":
        if "tool_calls" in truncated_messages[-1]:
            del truncated_messages[-1]["tool_calls"]
        # If the message is now empty (no content), remove the entire message
        if not truncated_messages[-1].get("content") and not truncated_messages[-1].get("tool_calls"):
            truncated_messages.pop()
    
    # Create expected output with the target tool call
    expected_output = {
        "role": "assistant",
        "tool_calls": [target_tool_call]
    }
    
    return truncated_messages, expected_output

def expand_conversation(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a single conversation item into multiple samples for intermediate tool calls.
    
    Args:
        item: Original conversation item in eval format
        
    Returns:
        List of expanded conversation items (includes original + intermediate samples)
    """
    # Extract messages - handle both direct format and wrapped format
    messages = item.get("messages")
    if messages is None and "item" in item:
        messages = item["item"].get("messages")
    
    if not messages:
        return [item]  # Return original if no messages found
    
    # Find all tool calls in the conversation
    tool_calls = extract_tool_calls_from_messages(messages)
    
    if not tool_calls:
        return [item]  # Return original if no tool calls found
    
    expanded_items = []
    
    # Create one sample for each tool call (intermediate samples)
    for message_index, tool_call in tool_calls:
        # Create truncated conversation up to this tool call
        truncated_messages, expected_output = create_truncated_conversation(
            messages, message_index, tool_call
        )
        
        # Create new item with truncated messages and expected output
        new_item = copy.deepcopy(item)
        
        # Handle both direct format and wrapped format
        if "item" in new_item:
            new_item["item"]["messages"] = truncated_messages
            new_item["item"]["expected_output"] = expected_output
        else:
            new_item["messages"] = truncated_messages
            new_item["expected_output"] = expected_output
        
        expanded_items.append(new_item)
    
    # Always include the original complete conversation as well
    expanded_items.append(item)
    
    return expanded_items

def analyze_tool_calls(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Analyze tool call distribution in a list of items.
    
    Args:
        items: List of conversation items
        
    Returns:
        Dictionary with tool call counts
    """
    tool_counter = Counter()
    
    for item in items:
        # First check if this is an expanded item with expected_output
        expected_output = item.get("expected_output", {})
        if "tool_calls" in expected_output:
            for tool_call in expected_output["tool_calls"]:
                function_name = tool_call.get("function", {}).get("name", "unknown")
                tool_counter[function_name] += 1
        else:
            # Check if this is original format with item.expected_output
            if "item" in item and "expected_output" in item["item"]:
                expected_output = item["item"]["expected_output"]
                if "tool_calls" in expected_output:
                    for tool_call in expected_output["tool_calls"]:
                        function_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_counter[function_name] += 1
            else:
                # For original items, extract tool calls from messages
                messages = item.get("messages")
                if messages is None and "item" in item:
                    messages = item["item"].get("messages")
                
                if messages:
                    tool_calls = extract_tool_calls_from_messages(messages)
                    for _, tool_call in tool_calls:
                        function_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_counter[function_name] += 1
    
    return dict(tool_counter)

def analyze_conversation_depth(items: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Analyze conversation depth distribution.
    
    Args:
        items: List of conversation items
        
    Returns:
        Dictionary mapping depth to count
    """
    depth_counter = Counter()
    
    for item in items:
        messages = item.get("messages")
        if messages is None and "item" in item:
            messages = item["item"].get("messages")
        
        if messages:
            depth_counter[len(messages)] += 1
    
    return dict(depth_counter)

def main():
    # Resolve input and output file paths
    input_path = Path(INPUT_FILE)
    eval_output_path = input_path.parent / f"{input_path.stem}_eval{input_path.suffix}"
    expanded_output_path = input_path.parent / f"{input_path.stem}_eval_expanded{input_path.suffix}"
    
    print(f"📖 Reading from: {input_path}")
    print(f"💾 Will write eval format to: {eval_output_path}")
    print(f"💾 Will write expanded format to: {expanded_output_path}")
    
    if not input_path.exists():
        print(f"❌ Error: Input file {input_path} does not exist!")
        return
    
    # ========================================
    # STAGE 1: Convert SFT to Eval Format
    # ========================================
    print("\n" + "="*60)
    print("STAGE 1: Converting SFT to Eval Format")
    print("="*60)
    
    # Read SFT data
    sft_items = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                sft_items.append(item)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping malformed JSON on line {line_num}: {e}")
                continue
    
    print(f"📊 SFT samples read: {len(sft_items)}")
    
    # Convert to eval format
    eval_items = []
    skipped_count = 0
    for item in sft_items:
        eval_item = convert_sft_to_eval(item)
        if eval_item:
            eval_items.append(eval_item)
        else:
            skipped_count += 1
    
    print(f"✅ Converted to eval format: {len(eval_items)}")
    if skipped_count > 0:
        print(f"⏭️  Skipped (no tool calls): {skipped_count}")
    
    # Write eval format data
    with eval_output_path.open("w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"💾 Saved eval format to: {eval_output_path}")
    
    # ========================================
    # STAGE 2: Expand Eval Format
    # ========================================
    print("\n" + "="*60)
    print("STAGE 2: Expanding Eval Format")
    print("="*60)
    
    # Calculate max depth (highest number of tool calls in any conversation)
    max_tool_calls_per_conversation = 0
    for item in eval_items:
        messages = item["item"]["messages"]
        tool_calls = extract_tool_calls_from_messages(messages)
        max_tool_calls_per_conversation = max(max_tool_calls_per_conversation, len(tool_calls))
    
    # Expand all conversations
    expanded_items = []
    for item in eval_items:
        expanded_items.extend(expand_conversation(item))
    
    print(f"📊 Eval samples: {len(eval_items)}")
    print(f"🔄 Expanded samples: {len(expanded_items)}")
    print(f"📈 Expansion ratio: {len(expanded_items) / len(eval_items):.2f}x")
    print(f"🏔️  Highest conversation depth: {max_tool_calls_per_conversation} tool calls")
    
    # Write expanded data
    with expanded_output_path.open("w", encoding="utf-8") as f:
        for item in expanded_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Expansion complete! Saved to: {expanded_output_path}")
    
    # ========================================
    # Statistics
    # ========================================
    print("\n" + "="*60)
    print("📊 CONVERSION & EXPANSION STATISTICS")
    print("="*60)
    
    # Analyze data
    eval_tool_calls = analyze_tool_calls(eval_items)
    expanded_tool_calls = analyze_tool_calls(expanded_items)
    eval_depth_dist = analyze_conversation_depth(eval_items)
    expanded_depth_dist = analyze_conversation_depth(expanded_items)
    
    print(f"\n📝 Sample Counts:")
    print(f"   Original SFT samples:     {len(sft_items):,}")
    print(f"   Converted eval samples:   {len(eval_items):,}")
    print(f"   Expanded samples:         {len(expanded_items):,}")
    print(f"   Skipped samples:          {skipped_count:,}")
    print(f"   Expansion factor:         {len(expanded_items) / len(eval_items):.2f}x")
    print(f"   Max depth reached:        {max_tool_calls_per_conversation} tool calls")
    
    print(f"\n🛠️  Tool Call Distribution - EVAL Format:")
    for tool, count in sorted(eval_tool_calls.items()):
        print(f"   {tool:<30} {count:>6,}")
    print(f"   {'TOTAL':<30} {sum(eval_tool_calls.values()):>6,}")
    
    print(f"\n🛠️  Tool Call Distribution - EXPANDED Format:")
    for tool, count in sorted(expanded_tool_calls.items()):
        print(f"   {tool:<30} {count:>6,}")
    print(f"   {'TOTAL':<30} {sum(expanded_tool_calls.values()):>6,}")
    
    print(f"\n📏 Conversation Length Distribution - EVAL Format:")
    for depth in sorted(eval_depth_dist.keys()):
        print(f"   {depth:>2} messages: {eval_depth_dist[depth]:>6,} conversations")
    
    print(f"\n📏 Conversation Length Distribution - EXPANDED Format:")
    for depth in sorted(expanded_depth_dist.keys()):
        print(f"   {depth:>2} messages: {expanded_depth_dist[depth]:>6,} conversations")
    
    print("\n" + "="*60)
    print("✅ ALL DONE!")
    print("="*60)
    print(f"📁 Output files:")
    print(f"   1. {eval_output_path}")
    print(f"   2. {expanded_output_path}")

if __name__ == "__main__":
    main()
