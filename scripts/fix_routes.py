import sys

path = r'c:\Users\DELL\Downloads\rtrp (1)\rtrp\backend\routes\pipeline_routes.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if '@pipeline_bp.route("/upload", methods=["POST"])' in line:
        new_lines.append(line)
        new_lines.append('def upload_csv():\n')
        new_lines.append('    """Unified persistent upload handler."""\n')
        new_lines.append('    try:\n')
        new_lines.append('        from controllers.product_intelligence_controller import handle_product_upload\n')
        new_lines.append('        return handle_product_upload(request)\n')
        new_lines.append('    except Exception as e:\n')
        new_lines.append('        logger.error("Persistent upload failed: %s", e)\n')
        new_lines.append('        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500\n')
        skip = True
        continue
    
    # We want to skip everything until the next route
    if skip:
        if '@pipeline_bp.route' in line and i > 0 and 'upload' not in line:
            skip = False
        else:
            continue
    
    new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("File fixed successfully")
