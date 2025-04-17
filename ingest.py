from docling.document_converter import DocumentConverter

source = "data/insiel.pdf"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())

markdown_text = result.document.export_to_markdown()
image_count = markdown_text.count("<!-- image -->")

print(f"🔢 Immagini trovate: {image_count}")