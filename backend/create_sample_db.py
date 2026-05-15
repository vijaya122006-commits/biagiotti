import sqlite3
import os

db_path = r'c:\Users\DELL\Downloads\myrtrp 6 (1)\myrtrp 6\biagiotti\backend\sample_store.sqlite'

# Remove if exists to start fresh
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create products table
cursor.execute('''
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT,
    brand TEXT,
    category TEXT,
    price REAL,
    stock_quantity INTEGER,
    ingredients TEXT
)
''')

# Create sales table
cursor.execute('''
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT,
    units_sold INTEGER,
    sale_date TEXT,
    revenue REAL
)
''')

# Insert sample products
products = [
    ('P101', 'Vitamin C Serum', 'Biagiotti Glow', 'Skincare', 1250.0, 45, 'Aqua, Vitamin C, Hyaluronic Acid'),
    ('P102', 'Matte Lipstick (Ruby)', 'Biagiotti Color', 'Makeup', 850.0, 12, 'Wax, Pigments, Vitamin E'),
    ('P103', 'Hydrating Cleanser', 'Pure Essence', 'Skincare', 650.0, 80, 'Aloe Vera, Glycerin, Water'),
    ('P104', 'Sunscreen SPF 50', 'Safe Shield', 'Skincare', 950.0, 5, 'Zinc Oxide, Avobenzone'),
]

cursor.executemany('INSERT INTO products VALUES (?,?,?,?,?,?,?)', products)

# Insert sample sales
sales = [
    ('P101', 5, '2026-05-10', 6250.0),
    ('P101', 3, '2026-05-11', 3750.0),
    ('P102', 10, '2026-05-12', 8500.0),
    ('P104', 2, '2026-05-12', 1900.0),
]

cursor.executemany('INSERT INTO sales (product_id, units_sold, sale_date, revenue) VALUES (?,?,?,?)', sales)

conn.commit()
conn.close()

print(f"DONE: SQLite database created at: {db_path}")
