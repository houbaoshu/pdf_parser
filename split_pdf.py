import os
from pypdf import PdfReader, PdfWriter

def split_pdf(input_path, pages_per_part=10):
    if not os.path.exists(input_path):
        print(f"错误：找不到文件{input_path}")
        return
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)    
    file_name = os.path.splitext(input_path)[0]

    for i in range(0, total_pages, pages_per_part):
        writer = PdfWriter()
        end_page = min(i + pages_per_part, total_pages)

        for page_num in range(i, end_page):
            writer.add_page(reader.pages[page_num])
            
        output_filename = f"{file_name}_part_{i//pages_per_part + 1}.pdf"
        with open(output_filename, "wb") as output_pdf:
            writer.write(output_pdf)
        print(f"Created: {output_filename}")

split_pdf("1.pdf", pages_per_part=10)




            




