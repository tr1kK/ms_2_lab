import numpy
import matplotlib.pyplot as plt
import fitz
import pylatex


variant = 45
exp_pdf = "distribution_data/MS_D_Exp.pdf"
norm_pdf = "distribution_data/MS_D_Norm.pdf"
uniform_pdf = "distribution_data/MS_D_Unif.pdf"


def read_array_pdf(v, filename, flag=False):
    with fitz.Document(filename) as doc:
        for page in doc.pages():
            if page.search_for(f"Вариант\n{v}"):
                page_text = page.get_text()
                page_text = page_text.replace(",", ".").split("\n")
                k = page_text.index(f"{v}") + 1
                if flag:
                    a = float(page_text[k].split(" ")[1])
                    b = float(page_text[k + 1].split(" ")[1])
                    k += 2
                    return a, b, list(map(float, page_text[k:k + 200]))
                return list(map(float, page_text[k:k + 200]))


def create_table():
    doc = pylatex.Document("test")
    doc.append("Some text")
    doc.generate_pdf()


if __name__ == "__main__":
    exp = read_array_pdf(variant, exp_pdf)
    norm = read_array_pdf(variant, norm_pdf)
    a_uniform, b_uniform, uniform = read_array_pdf(variant, uniform_pdf, flag=True)
    lamb = 1.275
    m = 8
    create_table()
