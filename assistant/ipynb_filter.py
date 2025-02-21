#!/usr/bin/env python3

# pandoc --from ipynb --to markdown -o output.py ./input.ipynb --filter ./ipynb_filter.py

import panflute as pf
import pypandoc


def comment(elem):
    markdown_output = pf.convert_text(
        elem, input_format="panflute", output_format="markdown"
    )

    lines = markdown_output.split("\n")

    new_text = "\n".join(["# " + x for x in lines])

    return pf.RawBlock(new_text)


def process_divs(elem, doc):

    if isinstance(elem, pf.Div):

        classes = set(elem.classes)

        # Comment .markdown divs, keeping content
        if "markdown" in classes:
            return [comment(block) for block in elem.content]

        # Unnest .code divs
        if "code" in classes:
            return pf.RawBlock(pf.stringify(elem))

        # Remove .output classes
        if "output" in classes:
            return []

    return elem


def main(doc=None):
    return pf.run_filter(process_divs, doc=doc)


def convert_ipynb(input_file):
    return pypandoc.convert_file(input_file, "markdown", filters=[__file__])


if __name__ == "__main__":
    main()
