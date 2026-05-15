# controllers/upload_controller.py
from utils.response_builder import build_success_response, build_error_response
from utils.file_validator import validate_sales_csv
import pandas as pd

def handle_upload(request):
    if 'file' not in request.files:
        return build_error_response("No file part")

    file = request.files['file']
    if file.filename == '':
        return build_error_response("No selected file")

    is_valid, message = validate_sales_csv(file)
    if not is_valid:
        return build_error_response(message)

    # Bug 2 fix: actually read and persist the uploaded file
    dealer_id = getattr(request, 'dealer_id', None)
    if not dealer_id:
        return build_error_response("Authentication required", status=401)

    try:
        file.seek(0)  # Reset stream after validation read
        df = pd.read_csv(file)

        # Normalise column names
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Detect file type by columns present
        is_sales = all(c in df.columns for c in ['product_id', 'units_sold'])
        is_products = all(c in df.columns for c in ['product_id', 'product_name'])

        if not is_sales and not is_products:
            return build_error_response(
                "CSV must contain either (product_id, units_sold) for sales "
                "or (product_id, product_name) for products."
            )

        from database.models import db, Sale, Product
        from datetime import datetime
        rows_added = 0

        if is_sales:
            for _, row in df.iterrows():
                try:
                    sale = Sale(
                        product_id=str(row['product_id']),
                        dealer_id=dealer_id,
                        units_sold=float(row.get('units_sold', 0)),
                        revenue=float(row.get('revenue', 0)),
                        year=int(row.get('year', datetime.utcnow().year)),
                        month=int(row.get('month', datetime.utcnow().month)),
                        region=str(row.get('region', '')),
                        brand=str(row.get('brand', '')),
                    )
                    db.session.add(sale)
                    rows_added += 1
                except Exception:
                    continue
            db.session.commit()

        elif is_products:
            for _, row in df.iterrows():
                try:
                    # Upsert: update if exists, insert if not
                    existing = Product.query.filter_by(
                        product_id=str(row['product_id']),
                        dealer_id=dealer_id
                    ).first()
                    if existing:
                        existing.product_name = str(row.get('product_name', existing.product_name))
                        existing.price = float(row['price']) if 'price' in row else existing.price
                        existing.ingredients = str(row.get('ingredients', existing.ingredients or ''))
                        existing.category = str(row.get('category', existing.category or ''))
                        existing.brand = str(row.get('brand', existing.brand or ''))
                    else:
                        product = Product(
                            product_id=str(row['product_id']),
                            dealer_id=dealer_id,
                            product_name=str(row['product_name']),
                            brand=str(row.get('brand', '')),
                            category=str(row.get('category', '')),
                            price=float(row['price']) if 'price' in row else None,
                            ingredients=str(row.get('ingredients', '')),
                            skin_suitability=str(row.get('skin_suitability', 'all')),
                        )
                        db.session.add(product)
                    rows_added += 1
                except Exception:
                    continue
            db.session.commit()

        return build_success_response(
            data={"rows_imported": rows_added, "type": "sales" if is_sales else "products"},
            message=f"Successfully imported {rows_added} rows."
        )

    except Exception as e:
        from database.models import db
        db.session.rollback()
        return build_error_response(str(e))

