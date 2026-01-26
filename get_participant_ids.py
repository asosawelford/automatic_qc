import csv

# --- Configuration ---
input_file = "usable_audios_combined_check.txt"      # your original file
output_file = "patients_impact.csv" # filtered output file

# --- Read lines ---
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

rows = []

for line in lines:
    # Step 1: Trim at first "__"
    trimmed = line.split("__")[0]

    # Step 2: Skip anything with "_A"
    if "_A" in trimmed:
        continue

    # Step 3: Split into two parts
    if "_" in trimmed:
        parts = trimmed.split("_", 1)
        if len(parts) == 2:
            site, patient = parts

            # Step 4: Skip patients ending with 'a', 'A', or a digit
            if patient[-1].lower() == "a" or patient[-1].isdigit():
                continue

            rows.append([site, patient])

# --- Save to CSV ---
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Site", "Patient"])  # keep header
    writer.writerows(rows)

print(f"Saved {len(rows)} filtered entries to {output_file}")
