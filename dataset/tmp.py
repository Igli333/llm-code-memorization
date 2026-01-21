import pandas as pd
import re
from pathlib import Path

def extract_dates(text):
    """Extract date patterns from text in various contexts"""
    dates_found = []
    
    # Common date patterns with context
    date_patterns = [
        (r'"(\d{4}-\d{2}-\d{2})"', 'quoted'),  # "YYYY-MM-DD"
        (r'"(\d{2}-\d{2}-\d{4})"', 'quoted'),  # "DD-MM-YYYY"
        (r'"(\d{4}/\d{2}/\d{2})"', 'quoted'),  # "YYYY/MM/DD"
        (r'"(\d{2}/\d{2}/\d{4})"', 'quoted'),  # "DD/MM/YYYY"
        (r'/(\d{4}-\d{2}-\d{2})/', 'path'),    # /YYYY-MM-DD/ in paths
        (r'/(\d{2}-\d{2}-\d{4})/', 'path'),    # /DD-MM-YYYY/ in paths
    ]
    
    for pattern, context in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            dates_found.append({
                'date': match.group(1),
                'context': context,
                'full_match': match.group(0)
            })
    
    return dates_found

def has_simple_exception(text):
    """Check if line has exactly one exception with a simple string"""
    # Find all throw new statements
    exception_pattern = r'throw new \w+\((.*?)\);'
    matches = re.findall(exception_pattern, text)
    
    if len(matches) != 1:
        return None, None
    
    exception_content = matches[0]
    
    # Check if it's a simple string (only one pair of quotes, no +, no variables)
    quote_count = exception_content.count('"')
    
    if quote_count == 2 and '+' not in exception_content:
        # Extract the string content
        string_match = re.search(r'"([^"]+)"', exception_content)
        if string_match:
            return string_match.group(1), exception_content
    
    return None, None

def extract_parameters(text):
    """Extract function parameters with at least 8 characters"""
    # Match function signature parameters
    param_pattern = r'\(([^)]+)\)'
    match = re.search(param_pattern, text)
    
    if not match:
        return []
    
    params_str = match.group(1)
    
    # Split by comma and extract parameter names
    params = []
    for param in params_str.split(','):
        param = param.strip()
        if param:
            # Get the parameter name (last word)
            parts = param.split()
            if parts:
                param_name = parts[-1]
                # Check if it's at least 8 characters
                if len(param_name) >= 8:
                    params.append(param_name)
    
    return params

def mask_date(text, date_info):
    """Mask a date in the text based on its context"""
    if date_info['context'] == 'quoted':
        # Replace "date" with <DATE_OBJECT>
        masked = text.replace(f'"{date_info["date"]}"', '<DATE_OBJECT>')
    elif date_info['context'] == 'path':
        # Replace /date/ with /<DATE_OBJECT>/
        masked = text.replace(f'/{date_info["date"]}/', '/<DATE_OBJECT>/')
    return masked

def mask_exception(text, exception_str):
    """Mask an exception string in the text"""
    masked = text.replace(f'"{exception_str}"', '<EXCEPTION_STRING>')
    return masked

def mask_variable(text, var_name):
    """Mask all occurrences of a variable in the text"""
    # Use word boundary to avoid partial matches
    masked = re.sub(r'\b' + re.escape(var_name) + r'\b', '<VARIABLE>', text)
    return masked

def process_files(cs_file, java_file, output_csv):
    # Read the files
    with open(cs_file, 'r', encoding='utf-8') as f:
        cs_lines = f.readlines()
    
    with open(java_file, 'r', encoding='utf-8') as f:
        java_lines = f.readlines()
    
    # Storage for results
    results = []
    seen_dates = set()
    seen_exceptions = set()
    seen_variables = set()
    
    date_count = 0
    exception_count = 0
    variable_count = 0
    
    max_per_group = 50
    
    print("Processing files...")
    print(f"Total lines: {len(java_lines)}")
    
    # Process DATE objects first
    print("\n=== Processing DATE objects ===")
    for idx, (cs_line, java_line) in enumerate(zip(cs_lines, java_lines), start=1):
        if date_count >= max_per_group:
            break
        
        cs_line = cs_line.strip()
        java_line = java_line.strip()
        
        # Extract dates from Java line
        java_dates = extract_dates(java_line)
        
        for date_info in java_dates:
            date_str = date_info['date']
            if date_str not in seen_dates and date_count < max_per_group:
                # Check if date exists in CS line with same context
                cs_dates = extract_dates(cs_line)
                cs_date_match = None
                
                for cs_date_info in cs_dates:
                    if cs_date_info['date'] == date_str and cs_date_info['context'] == date_info['context']:
                        cs_date_match = cs_date_info
                        break
                
                if cs_date_match:
                    java_masked = mask_date(java_line, date_info)
                    cs_masked = mask_date(cs_line, cs_date_match)
                    
                    results.append({
                        'idx': idx,
                        'group': 'DATE',
                        'cs_raw': cs_line,
                        'cs_masked': cs_masked,
                        'cs_secret': date_str,
                        'java_raw': java_line,
                        'java_masked': java_masked,
                        'java_secret': date_str
                    })
                    
                    seen_dates.add(date_str)
                    date_count += 1
                    print(f"Found DATE #{date_count}: {date_str} ({date_info['context']}) at line {idx}")
                    break
    
    print(f"\nTotal DATE objects found: {date_count}")
    
    # Process EXCEPTION_STRING objects
    print("\n=== Processing EXCEPTION_STRING objects ===")
    for idx, (cs_line, java_line) in enumerate(zip(cs_lines, java_lines), start=1):
        if exception_count >= max_per_group:
            break
        
        cs_line = cs_line.strip()
        java_line = java_line.strip()
        
        # Check for simple exception in Java line
        exception_str, full_content = has_simple_exception(java_line)
        
        if exception_str and exception_str not in seen_exceptions:
            # Check if the same exception exists in CS line
            cs_exception_str, cs_full_content = has_simple_exception(cs_line)
            
            if cs_exception_str and exception_str == cs_exception_str:
                java_masked = mask_exception(java_line, exception_str)
                cs_masked = mask_exception(cs_line, cs_exception_str)
                
                results.append({
                    'idx': idx,
                    'group': 'EXCEPTION_STRING',
                    'cs_raw': cs_line,
                    'cs_masked': cs_masked,
                    'cs_secret': cs_exception_str,
                    'java_raw': java_line,
                    'java_masked': java_masked,
                    'java_secret': exception_str
                })
                
                seen_exceptions.add(exception_str)
                exception_count += 1
                print(f"Found EXCEPTION #{exception_count}: {exception_str[:50]}... at line {idx}")
    
    print(f"\nTotal EXCEPTION_STRING objects found: {exception_count}")
    
    # Process VARIABLE objects
    print("\n=== Processing VARIABLE objects ===")
    for idx, (cs_line, java_line) in enumerate(zip(cs_lines, java_lines), start=1):
        if variable_count >= max_per_group:
            break
        
        cs_line = cs_line.strip()
        java_line = java_line.strip()
        
        # Extract parameters from Java line
        params = extract_parameters(java_line)
        
        for param in params:
            if param not in seen_variables and variable_count < max_per_group:
                # Check if variable appears in the function body (not just signature)
                occurrences = len(re.findall(r'\b' + re.escape(param) + r'\b', java_line))
                
                if occurrences >= 1:
                    # Try to find corresponding variable in CS line
                    cs_params = extract_parameters(cs_line)
                    
                    cs_param = None
                    for cp in cs_params:
                        if len(cp) >= 8:
                            cs_param = cp
                            break
                    
                    if cs_param:
                        java_masked = mask_variable(java_line, param)
                        cs_masked = mask_variable(cs_line, cs_param)
                        
                        results.append({
                            'idx': idx,
                            'group': 'VARIABLE',
                            'cs_raw': cs_line,
                            'cs_masked': cs_masked,
                            'cs_secret': cs_param,
                            'java_raw': java_line,
                            'java_masked': java_masked,
                            'java_secret': param
                        })
                        
                        seen_variables.add(param)
                        variable_count += 1
                        print(f"Found VARIABLE #{variable_count}: {param} at line {idx}")
                        break
    
    print(f"\nTotal VARIABLE objects found: {variable_count}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n=== Summary ===")
    print(f"Total rows generated: {len(df)}")
    print(f"DATE: {date_count}")
    print(f"EXCEPTION_STRING: {exception_count}")
    print(f"VARIABLE: {variable_count}")
    print(f"\nCSV saved to: {output_csv}")
    
    return df

cs_file = 'train.java-cs.txt.cs'
java_file = 'train.java-cs.txt.java'
output_csv = 'masked_secrets.csv'

df = process_files(cs_file, java_file, output_csv)