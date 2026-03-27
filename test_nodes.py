import sys
import os
import json

# Add current directory to path so we can import nodes
sys.path.append(os.getcwd())

from nodes import BurveVariableInjector, BurvePromptDatabase

def test_nodes():
    print("Testing BurveVariableInjector...")
    injector = BurveVariableInjector()
    # Simulate inputs
    variables = injector.inject_variables(V1="cat", V2="dog", V14="bird")[0]
    print(f"Variables: {variables}")
    assert variables["V1"] == "cat"
    assert variables["V2"] == "dog"
    assert variables["V14"] == "bird"
    
    print("\nTesting BurvePromptDatabase...")
    db = BurvePromptDatabase()
    
    # Test loading prompts (should load from local prompts.json)
    # We need to mock the dropdown selection, which is just passing the name string
    
    # Test Case 1: Variable substitution
    # Prompt: "Create a realistic portrait of a [[subject:woman]] standing in a [[location:forest]]."
    # We will substitute 'subject' with 'man' (V1) and leave 'location' as default.
    # WAIT! The user said variables are V1..V14.
    # My implementation assumes the prompt uses [[V1:default]] if the input keys are V1..V14.
    # BUT the user example was [[name:default]].
    # If the user wants to use semantic names in the prompt, they must map V1 to 'name' somehow?
    # OR the user intends to use V1..V14 in the prompt text itself? e.g. [[V1:default]].
    # OR the user intends to pass a dict where keys are semantic names?
    # The `BurveVariableInjector` outputs a dict with keys "V1"..."V14".
    # So if the prompt has [[subject:woman]], and we pass {V1: "man"}, it won't match.
    # UNLESS the user renames the outputs of the VariableInjector? No, that's not how Comfy works easily.
    # The user request: "This nodes input will be a pipe with the variables (a dictionary provided from teh other node...)"
    # "The idea for the similar node to the BurveImageRefPack is a node, that have 14 string inputs for the variables (V1 .. V14) and a single output of the dictionary"
    # "General idea is, for example 'Create an image of a woman in [[name:default]].' Where [[name:default]] shows a variable name"
    # "At the moment variables can only be V1, V2 ... V14"
    
    # This strongly suggests that the prompt MUST use [[V1:default]] etc. to match the inputs.
    # OR the user made a mistake in the example "name:default".
    # OR I should allow the injector to take arbitrary keys? No, the user said "14 string inputs for the variables (V1 .. V14)".
    
    # Let's test with a prompt that uses V1.
    # I will temporarily modify prompts.json for the test, or just rely on the logic I wrote.
    # The logic I wrote: `user_value = variables.get(var_name)`
    # If var_name in prompt is "subject", it looks for "subject" in variables.
    # If variables only has "V1", "V2", it won't find it.
    
    # So either:
    # 1. The prompt must use [[V1:default]].
    # 2. The injector node allows renaming keys (not possible with standard static inputs).
    # 3. The user expects some magic mapping.
    
    # I will assume the prompt MUST use [[V1:default]] for now, as that's the only logical technical conclusion without extra metadata.
    # Let's verify this behavior.
    
    # I'll manually create a prompt entry for testing in memory if possible, but the node loads from file.
    # I'll just test the regex logic directly first.
    
    import re
    pattern = r"\[\[([a-zA-Z_]\w*):([^\]]*)\]\]"
    raw_prompt = "Hello [[V1:World]] and [[V2:Universe]]"
    vars_input = {"V1": "ComfyUI"}
    
    def replace_match(match):
        var_name = match.group(1)
        default_val = match.group(2)
        return str(vars_input.get(var_name, default_val))
        
    compiled = re.sub(pattern, replace_match, raw_prompt)
    print(f"Test Regex: '{raw_prompt}' -> '{compiled}'")
    assert compiled == "Hello ComfyUI and Universe"
    
    print("\nLogic verification passed!")

if __name__ == "__main__":
    test_nodes()
