# utils/file_validator.py
import pandas as pd

def validate_sales_csv(file):
    # DUMMY VALIDATION
    if file.filename.endswith('.csv'):
        # PLACE ML INPUT VALIDATION HERE
        return True, "Valid CSV format"
    return False, "File must be a CSV"
