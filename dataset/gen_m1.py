import pandas as pd

def process_files(java_file, cs_file, output_csv):
    rows = []
    row_id = 1

    with open(java_file, encoding="utf-8") as f:
        java_lines = f.readlines()

    with open(cs_file, encoding="utf-8") as f:
        cs_lines = f.readlines()

    for java_line, cs_line in zip(java_lines, cs_lines):
        rows.append({
            "id": row_id,
            "java_code": java_line.rstrip("\n"),
            "cs_code": cs_line.rstrip("\n")
        })
        row_id += 1

    df = pd.DataFrame(rows)
    df = df[["id", "java_code", "cs_code"]]
    df.to_csv(output_csv, index=False)
    return df

# -------------------------
# Run
# -------------------------
java_file = "train.java-cs.txt.java"
cs_file = "train.java-cs.txt.cs"
output_csv = "paired_codes.csv"

df = process_files(java_file, cs_file, output_csv)
print(df.head())
