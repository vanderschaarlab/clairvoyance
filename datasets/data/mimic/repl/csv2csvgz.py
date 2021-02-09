"""Convert csv to csv.gz.
"""

# Necessary packages
import gzip

for filename in [
    "mimic_static_test_data.csv",
    "mimic_static_train_data.csv",
    "mimic_temporal_test_data_eav.csv",
    "mimic_temporal_train_data_eav.csv",
]:
    with open(filename, "rb") as f_in:
        with gzip.open(filename + ".gz", "wb") as f_out:
            f_out.writelines(f_in)
