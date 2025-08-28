#!/usr/bin/env python3
"""Simple syntax test for analysis.py"""

try:
    # Try to import the module
    import ast
    with open('drl/utils/analysis.py', 'r') as f:
        source = f.read()
    
    # Parse the AST
    ast.parse(source)
    print("✅ Syntax check passed! The file has valid Python syntax.")
    
    # Try to import specific classes
    exec(source)
    print("✅ Code execution test passed! All classes and functions are properly defined.")
    
except SyntaxError as e:
    print(f"❌ Syntax error found: {e}")
    print(f"   Line {e.lineno}: {e.text}")
except Exception as e:
    print(f"❌ Other error found: {e}")
    import traceback
    traceback.print_exc()

