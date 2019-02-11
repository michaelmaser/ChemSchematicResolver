# -*- coding: utf-8 -*-
"""
test_ocr
========

Test optical character recognition

"""

import unittest
import os
import chemschematicdiagramextractor as csde


tests_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(os.path.dirname(tests_dir), 'train')
examples_dir = os.path.join(train_dir, 'train_imgs')

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

class TestOcr(unittest.TestCase):

    def test_get_lines(self):
        pass

    def test_ocr_whole_image(self):
        '''
        Uses the OCR module on the whole image to identify text blocks
        :return:
        '''

        test_imgs = [os.path.join(examples_dir, file) for file in os.listdir(examples_dir)]

        for img_path in test_imgs:
            fig = csde.io.imread(img_path)  # Read in float and raw pixel images
            text_blocks = csde.ocr.get_text(fig.img)

            # Create output image
            out_fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(fig.img)

            for text_block in text_blocks:

                diag_rect = mpatches.Rectangle((text_block.left, text_block.top), text_block.width, text_block.height,
                                               fill=False, edgecolor='r', linewidth=2)
                ax.add_patch(diag_rect)

            plt.show()


