"""
Instruction data formatting utilities for Weird AI.
"""


def format_input(entry):
    """
    Format one instruction example using an Alpaca-style prompt.
    This function does NOT include the response.
    """

    # TODO:
    # 1. Build the standard instruction text.
    # 2. Add the ### Instruction section.
    # 3. Add the ### Input section only when entry["input"] is not empty.
    # 4. Return the complete prompt.
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


def format_response(entry):
    """
    Format the expected response section.
    """

    # TODO:
    # Return a string like:
    # "\n\n### Response:\n..."
    desired_response = f"\n\n### Response:\n{entry['output']}"
    return desired_response


def format_full_example(entry):
    """
    Format an entire instruction-response example.
    """

    # TODO:
    # Combine format_input(entry) and format_response(entry).
    return format_input(entry) + format_response(entry)



def validate_instruction_entry(entry):
    """
    Validate that an instruction dataset entry has instruction, input, and output fields.
    """

    # TODO:
    # Check for instruction, input, and output keys.
    # Verify that instruction and output are not empty.

    if "instruction" not in entry or not entry["instruction"]:
        return False

    if "output" not in entry or not entry["output"]:
        return False

    if "input" not in entry:
        return False

    return True
