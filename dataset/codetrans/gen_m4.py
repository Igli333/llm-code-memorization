import pandas as pd
import random
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tsc_sharp
random.seed(42)

# -------------------------
# Tree-sitter setup
# -------------------------
JAVA_LANGUAGE = Language(tsjava.language())
C_SHARP_LANGUAGE = Language(tsc_sharp.language())

java_parser = Parser()
java_parser.language = JAVA_LANGUAGE

csharp_parser = Parser()
csharp_parser.language = C_SHARP_LANGUAGE


# -------------------------
# AST utilities
# -------------------------
TARGET_TYPES = {"expression_statement", "string_literal"}

def extract_statements(node, results):
    if node.type in TARGET_TYPES:
        results.append(node)

    for child in node.children:
        extract_statements(child, results)


def mask_node_in_code(code: str, node):
    """Replace node span with <MASKED>"""
    start = node.start_byte
    end = node.end_byte
    masked_fragment = code[start:end]
    masked_code = code[:start] + "<MASKED>" + code[end:]
    return masked_code, masked_fragment


def process_single_line(code, parser, lang):
    tree = parser.parse(code.encode("utf-8"))
    nodes = []
    extract_statements(tree.root_node, nodes)

    if not nodes:
        return None

    chosen = random.choice(nodes)
    code_masked, masked = mask_node_in_code(code, chosen)

    return {
        "lang": lang,
        "type": chosen.type,
        "code_raw": code.rstrip("\n"),
        "code_masked": code_masked.rstrip("\n"),
        "masked": masked.decode("utf-8") if isinstance(masked, bytes) else masked,
    }


# -------------------------
# Main processing
# -------------------------
def process_files(csv_file, output_csv):
    rows = []
    row_id = 1

    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        java_code = row['java_code']
        cs_code = row['cs_code']

        java_result = process_single_line(
            java_code, java_parser, "java"
        )
        if java_result:
            java_result["id"] = row_id
            rows.append(java_result)
            row_id += 1

        cs_result = process_single_line(
            cs_code, csharp_parser, "c#"
        )
        if cs_result:
            cs_result["id"] = row_id
            rows.append(cs_result)
            row_id += 1
    df_out = pd.DataFrame(rows)
    df_out = df_out[["id", "lang", "type", "code_raw", "code_masked", "masked"]]
    df_out.to_csv(output_csv, index=False)
    return df_out


# -------------------------
# Run
# -------------------------
csv_file = "paired_codes_sampled.csv"
output_csv = "masked_secrets.csv"

df = process_files(csv_file, output_csv)
print(df.head())
