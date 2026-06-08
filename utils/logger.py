"""Lightweight CSV / JSON logging helpers."""

import csv
import json


def write_dict_to_json(file_json, dict_data):
    with open(file_json, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict_data, indent=4))


def load_dict_from_json(file_json):
    with open(file_json) as fh:
        return json.load(fh)


class CSVWriter:
    """Append rows of training/evaluation metrics to a CSV file."""

    def __init__(self, file_name, column_names):
        self.file_name = file_name
        self.column_names = column_names
        self.file_handle = open(self.file_name, "w")
        self.writer = csv.writer(self.file_handle)
        self.write_header()
        print(f"{self.file_name} created successfully with header row")

    def write_header(self):
        self.write_row(self.column_names)

    def write_row(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file_handle.close()
