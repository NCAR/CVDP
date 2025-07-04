import matplotlib.pyplot as plt
import base64
import nbformat
from io import BytesIO
from datetime import datetime


class CVDPNotebook():
    def create_section(self, section_name, label=None, rank=1, label_hidden=False):
        if label_hidden:
            label = None
        elif label is None:
            label = section_name

        self.__sections[section_name] = {
            "cells": [],
            "rank": rank,
            "label": label
        }
    
    
    def __init__(self):
        self.__sections = {}
        self.__num_figs = 0

    
    def add_markdown_cell(self, cell_data, section_name):
        if section_name not in self.__sections:
            self.create_section(section_name)
        self.__sections[section_name]["cells"].append(nbformat.v4.new_markdown_cell(cell_data))
    
    
    def add_figure_cell(self, figure, section_name=None, alt_text="Figure"):
        img_buffer = BytesIO()
        figure.savefig(img_buffer, format='png')
        img_buffer.seek(0)
    
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        cell_data = f'![{alt_text}](data:image/png;base64,{img_base64})'
        img_buffer.close()
        
        self.add_markdown_cell(cell_data=cell_data, section_name=section_name)
        self.__num_figs += 1

    
    def set_section_label(section_name, section_label):
        self.__sections[section_name]["label"] = section_label
    

    def _format_section_label(self, label):
        return f"## {label}"
    
    
    def save_notebook(self, path="CVDP_output.ipynb", title=None):
        from cvdp.utils import get_version, get_time_stamp
        from cvdp.definitions import PATH_BANNER_PNG

        cell_data = ""
        with open(PATH_BANNER_PNG, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            header_img_data = f'![CVDP Banner](data:image/png;base64,{img_base64})'
        
        header_data = [
            header_img_data,
            '\n',
            '' if title is None else f'## Deck Title: {title}\n',
            '\n',
            '```\n',
            'Webpage: https://github.com/NCAR/CVDP/tree/main\n',
            f'Version: CVDP {get_version()}\n',
            f'Generation Timestamp: {get_time_stamp()}\n',
            f'Figures Generated: {self.__num_figs}\n',
            '```\n'
        ]

        self.create_section("header", rank=0, label_hidden=True)
        self.add_markdown_cell(header_data, "header")
        
        notebook_node = nbformat.v4.new_notebook()
        ranked_cells = {}
        for section_name in self.__sections:
            rank = self.__sections[section_name]["rank"]
            if rank in ranked_cells:
                ranked_cells[rank].append(section_name)
            else:
                ranked_cells[rank] = [section_name]

        ranks = list(ranked_cells.keys())
        ranks.sort()

        for rank in ranks:
            for section_name in ranked_cells[rank]:
                label = self.__sections[section_name]["label"]
                if label is not None:
                    label_cell = nbformat.v4.new_markdown_cell(self._format_section_label(label), metadata={"jp-MarkdownHeadingCollapsed": True})
                    
                    notebook_node.cells.append(label_cell)
                
                for cell in self.__sections[section_name]["cells"]:   
                    notebook_node.cells.append(cell)
        
        with open(path, 'w') as nb_file:
            nbformat.write(notebook_node, nb_file)