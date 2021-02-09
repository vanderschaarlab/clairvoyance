"""Convert csv.gz to csv.
"""

# Necessary packages
import gzip

for filename in [
    "mimic_static_test_data.csv.gz",
    "mimic_static_train_data.csv.gz",
    "mimic_temporal_test_data_eav.csv.gz",
    "mimic_temporal_train_data_eav.csv.gz",
]:
    with gzip.open(filename, "rt") as f:
        data = f.read()
        with open(filename[:-3], "wt") as f:
            f.write(data)
