from pathlib import Path

import pymupdf


def main() -> None:
    output = Path("examples/synthetic-chart.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    page = pdf.new_page()
    rows = [
        "SYNTHETIC CLINICAL NOTE",
        "MRN: 4471902",
        "DOB: 03/14/1962",
        "Date of service: 06/02/2024",
        "Phone: (415) 555-0138",
        "SSN: 123-45-6789",
        "Clinical content: shortness of breath and ankle swelling.",
    ]
    for index, row in enumerate(rows):
        page.insert_text((72, 90 + index * 30), row, fontsize=12)
    pdf.save(output)
    pdf.close()
    print(output)


if __name__ == "__main__":
    main()
