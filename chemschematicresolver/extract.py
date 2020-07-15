# -*- coding: utf-8 -*-
"""
Extract
=======

Functions to extract diagram-label pairs from schematic chemical diagrams.

author: Ed Beard
email: ejb207@cam.ac.uk

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

from .io import imread
from .actions import segment, classify_kmeans, preprocessing, label_diags, read_diagram_pyosra
from .clean import clean_output
from .ocr import read_label
from .r_group import detect_r_group, get_rgroup_smiles
from .validate import is_false_positive, remove_repeating

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
import urllib
import math
import tarfile, zipfile
# mmaser adding import json for extract_reaction_batch output format
import json

from chemdataextractor import Document

log = logging.getLogger(__name__)


def extract_document(filename, extract_all=True, allow_wildcards=False, output=os.path.join(os.path.dirname(os.getcwd()), 'csd')):
    """ Extracts chemical records from a document and identifies chemical schematic diagrams.
    Then substitutes in if the label was found in a record

    :param filename: Location of document to be extracted
    :param extract_all : Boolean to determine whether output is combined with chemical records
    :param allow_wildcards: Bool to indicate whether results containing wildcards are permitted
    :param output: Directory to store extracted images

    :return : Dictionary of chemical records with diagram SMILES strings, or List of label candidates and smiles
    """

    log.info('Extracting from %s ...' % filename)

    # Extract the raw records from CDE
    doc = Document.from_file(filename)
    figs = doc.figures

    # Identify image candidates
    csds = find_image_candidates(figs, filename)

    # Download figures locally
    fig_paths = download_figs(csds, output)
    log.info("All relevant figures from %s downloaded successfully" % filename)

    # When diagrams are not found, return results without CSR extraction
    if extract_all and not fig_paths:
        log.info('No chemical diagrams detected. Returning chemical records.')
        return doc.records.serialize()
    elif not extract_all and not fig_paths:
        log.info('No chemical diagrams detected. Returning empty list.')
        return []

    log.info('Chemical diagram(s) detected. Running ChemSchematicResolver...')
    # Run CSR
    results = []
    for path in fig_paths:
        try:
            results.append(extract_image(path, allow_wildcards=allow_wildcards))
        except:
            log.error('Could not extract image at %s' % path)
            pass

    if not extract_all:
        return results

    records = doc.records.serialize()

    # Substitute smiles for labels
    combined_results = substitute_labels(records, results)
    log.info('All diagram results extracted and combined with chemical records.')

    return combined_results


def extract_documents(dirname, extract_all=True, allow_wildcards=False, output=os.path.join(os.path.dirname(os.getcwd()), 'csd')):
    """ Automatically identifies and extracts chemical schematic diagrams from all files in a directory of documents.

    :param dirname: Location of directory, with corpus to be extracted
    :param extract_all : Boolean indicating whether to extract all results (even those without chemical diagrams)
    :param allow_wildcards: Bool to indicate whether results containing wildcards are permitted
    :param output: Directory to store extracted images

    :return results: List of chemical record objects, enriched with chemical diagram information
    """

    log.info('Extracting all documents at %s ...' % dirname)

    results = []

    if os.path.isdir(dirname):
        # Extract from all files in directory
        for file in os.listdir(dirname):
            results.append(extract_document(os.path.join(dirname, file), extract_all, allow_wildcards, output))

    elif os.path.isfile(dirname):

        # Unzipping compressed inputs
        if dirname.endswith('zip'):
            # Logic to unzip the file locally
            log.info('Opening zip file...')
            zip_ref = zipfile.ZipFile(dirname)
            extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            zip_ref.extractall(extracted_path)
            zip_ref.close()

        elif dirname.endswith('tar.gz'):
            # Logic to unzip tarball locally
            log.info('Opening tarball file...')
            tar_ref = tarfile.open(dirname, 'r:gz')
            extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            tar_ref.extractall(extracted_path)
            tar_ref.close()

        elif dirname.endswith('tar'):
            # Logic to unzip tarball locally
            log.info('Opening tarball file...')
            tar_ref = tarfile.open(dirname, 'r:')
            extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            tar_ref.extractall(extracted_path)
            tar_ref.close()
        else:
            # Logic for wrong file type
            log.error('Input not a directory')
            raise NotADirectoryError

        docs = [os.path.join(extracted_path, doc) for doc in os.listdir(extracted_path)]
        for file in docs:
            results.append(extract_document(file, extract_all, allow_wildcards, output))

    return results


def substitute_labels(records, results):
    """ Looks for label candidates in the document records and substitutes where appropriate

    :param records: Serialized chemical records from chemdataextractor
    :param results: Results extracted from the diagram

    :returns: List of chemical records enriched with chemical diagram SMILES string
    """

    docs_labelled_records = []

    record_labels = [record for record in records if 'labels' in record.keys()]

    # Get all records that contain common labels
    for diag_result in results:
        for label_cands, smile in diag_result:
            for record_label in record_labels:
                overlap = [(record_label, label_cand, smile) for label_cand in label_cands if label_cand in record_label['labels']]
                docs_labelled_records += overlap

    log.debug(docs_labelled_records)

    # Adding data to the serialized ChemDataExtractor output
    for doc_record, diag_label, diag_smile in docs_labelled_records:
        for record in records:
            if record == doc_record:
                record['diagram'] = {'smiles': diag_smile, 'label': diag_label}

    return records


def download_figs(figs, output):
    """ Downloads figures from url

    :param figs: List of tuples in form figure metadata (Filename, figure id, url to figure, caption)
    :param output: Location of output images
    """

    if not os.path.exists(output):
        os.makedirs(output)

    fig_paths = []

    for file, id, url, caption in figs:

        img_format = url.split('.')[-1]
        log.info('Downloading %s image from %s' % (img_format, url))
        filename = file.split('/')[-1].rsplit('.', 1)[0] + '_' + id + '.' + img_format
        path = os.path.join(output, filename)

        log.debug("Downloading %s..." % filename)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path) # Saves downloaded image to file
        else:
            log.debug("File exists! Going to next image")

        fig_paths.append(path)

    return fig_paths


def find_image_candidates(figs, filename):
    """ Returns a list of csd figures

    :param figs: ChemDataExtractor figure objects
    :param filename: String of the file's name
    :return: List of figure metadata (Filename, figure id, url to figure, caption)
    :rtype:   list[tuple[string, string, string, string]]
    """
    csd_imgs = []

    for fig in figs:
        detected = False  # Used to avoid processing images twice
        records = fig.records
        caption = fig.caption
        for record in records:
            if detected:
                break

            rec = record.serialize()
            if 'figure' in rec.keys():
                detected = True
                log.info('Chemical schematic diagram instance found!')
                csd_imgs.append((filename, fig.id, fig.url, caption.text.replace('\n', ' ')))

    return csd_imgs

# mmaser: adding confidence threshold as arg to be adjustable
# TODO mmaser: try adding property .loc on smiles[i:j] in output for x_pos 
# (centroid) of structure and/or label to be able to assemble into reaction
#
def extract_image(filename, debug=False, allow_wildcards=False, 
                  confidence_threshold=None, count_false_positives=False):
    """ Converts a Figure containing chemical schematic diagrams to SMILES strings and extracted label candidates

    :param filename: Input file name for extraction
    :param debug: Bool to indicate debugging
    :param allow_wildcards: Bool to indicate whether results containing wildcards are permitted
    # mmaser added arg
    :param confidence_threshold: numerical in (0, 100) to determine tesserocr 
    cutoff for returning predicted labels/diagrams
    :param count_false_positives: Bool to indicate whether or not to include false positive count in the function return. Only useful with reaction_batch_extract.

    :return : List of label candidates and smiles
    :rtype : list[tuple[list[string],string]]
    """

    # Output lists
    r_smiles = []
    smiles = []

    extension = filename.split('.')[-1]

    # Confidence threshold for OCR results
    # mmaser: converting to conditional based on added arg
    if not confidence_threshold == None:   
        try:
            confidence_threshold = float(confidence_threshold)
        except:
            # set to default from original csr work
            confidence_threshold = 73.7620468139648
            log.debug('Unable to convert confidence_threshold to float, setting to default 73.762')
    else:
        log.info('No confidence_threshold provided, setting to default 73.762')
        confidence_threshold = 73.7620468139648

    # Read in float and raw pixel images
    fig = imread(filename)
    fig_bbox = fig.get_bounding_box()

    # Segment image into pixel islands
    # mmaser: returns list of bounding boxes as panel objects
    # can call self.tag on each panel object and sort to get ordered list
    # can also call self.centroid to get bbox center of each Panel 
    panels = segment(fig)

    # Initial classify of images, to account for merging in segmentation
    # mmaser: believe labels and diags have self.center property but is (x, y)
    labels, diags = classify_kmeans(panels, fig)

    # Preprocess image (eg merge labels that are small into larger labels)
    # mmaser: can still call self.tag on each label and diag in the lists  
    # outputted here (are lists of panel objects) and sort to get ordered list
    labels, diags = preprocessing(labels, diags, fig)

    # Re-cluster by height if there are more Diagram objects than Labels
    if len(labels) < len(diags):
        labels_h, diags_h = classify_kmeans(panels, fig, skel=False)
        labels_h, diags_h = preprocessing(labels_h, diags_h, fig)

        # Choose the fitting with the closest number of diagrams and labels
        if abs(len(labels_h) - len(diags_h)) < abs(len(labels) - len(diags)):
            labels = labels_h
            diags = diags_h

    if debug is True:
        # Create output image
        out_fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(fig.img)
        colours = iter(
            ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y', 'r', 'b', 'g', 'k', 'c', 'm', 'y'])

    # Add label information to the appropriate diagram by expanding bounding box
    ###### mmaser #######
    # may have to adjust return on label_diags to keep tag check output
    # YES, done. should now order by tag, which should track with x_position
    labelled_diags = label_diags(labels, diags, fig_bbox)
    labelled_diags = remove_repeating(labelled_diags)

    for diag in labelled_diags:

        label = diag.label

        if debug is True:

            colour = next(colours)

            # Add diag bbox to debug image
            diag_rect = mpatches.Rectangle((diag.left, diag.top), diag.width, diag.height,
                                           fill=False, edgecolor=colour, linewidth=2)
            ax.text(diag.left, diag.top + diag.height / 4, '[%s]' % diag.tag, size=diag.height / 20, color='r')
            ax.add_patch(diag_rect)

            # Add label bbox to debug image
            label_rect = mpatches.Rectangle((label.left, label.top), label.width, label.height,
                                            fill=False, edgecolor=colour, linewidth=2)
            ax.text(label.left, label.top + label.height / 4, '[%s]' % label.tag, size=label.height / 5, color='r')
            ax.add_patch(label_rect)

        ######## mmaser ########## start here 07/14/2020 to see what happens
        # with labels (and thus diagrams) under the confidence threshold.
        # read_label returns the same label with new property label.text 
        # where .text is a list of tagged Sentence objects which are  
        # constructed from the raw sentence list from a get_sentences call
        # on the text blocks returned from a get_text call on the image, 
        # which uses tesserocr to actually run the text prediction.
        # conf is also returned from the get_text call which is the average
        # confidence of the blocks returned by get_text, as calc'd by tesserocr
        #
        # Read the label
        diag.label, conf = read_label(fig, label)

        if not diag.label.text:
            log.warning('Text could not be resolved from label %s' % label.tag)

        # Only extract images where the confidence is sufficiently high
        ### mmaser: if this becomes problem with missing labels can adjust
        ### schema to still run pyosra on panels with diagram but skip label
        if not math.isnan(conf) and conf >= confidence_threshold:

            # mmaser: last chance get diag.center now
            #centroid = diag.center
            
            # Add r-group variables if detected
            diag = detect_r_group(diag)

            # Get SMILES for output
            ### mmaser: can also adjust call here for, e.g., get_graph
            ### if ever decide to output graphs directly instead of smiles
            smiles, r_smiles = get_smiles(diag, smiles, r_smiles, extension)

        else:
            log.warning('Confidence of label %s deemed too low for extraction' % diag.label.tag)

    log.info('The results are :')
    log.info('R-smiles %s' % r_smiles)
    log.info('Smiles %s' % smiles)
    if debug is True:
        ax.set_axis_off()
        plt.show()

    # mmaser: smiles should be ordered list by position now
    # mmaser: 07/14/2020 smiles and r_smiles are now 3-tuples
    # with (label, smiles, center(2-tuple))
    total_smiles = smiles + r_smiles

    # Removing false positives from lack of labels or wildcard smiles
    #output = [smile for smile in total_smiles if is_false_positive(smile, allow_wildcards=allow_wildcards) is False]
    
    # mmaser: add false_positive tag to be included with output
    false_positives = 0
    output = []
    for smile in total_smiles:
        if is_false_positive(smile, allow_wildcards=allow_wildcards) is False:
            output.append(smile)
        else:
            false_positives += 1
    
    if len(total_smiles) != len(output):
        log.warning('Some SMILES strings were determined to be false positives and were removed from the output.')
        log.info(f'{false_positives} false positives found.')

    log.info('Final Results : ')
    for result in output:
        log.info(result)

    if count_false_positives is True:
        return output, false_positives
        
    return output


def extract_images(dirname, debug=False, allow_wildcards=False):
    """ Extracts the chemical schematic diagrams from a directory of input images

    :param dirname: Location of directory, with figures to be extracted
    :param debug: Boolean specifying verbose debug mode.
    :param allow_wildcards: Bool to indicate whether results containing wildcards are permitted

    :return results: List of chemical record objects, enriched with chemical diagram information
    """

    log.info('Extracting all images at %s ...' % dirname)

    results = []

    if os.path.isdir(dirname):
        # Extract from all files in directory
        for file in os.listdir(dirname):
            results.append(extract_image(os.path.join(dirname, file), debug, allow_wildcards))

    elif os.path.isfile(dirname):

        # Unzipping compressed inputs
        if dirname.endswith('zip'):
            # Logic to unzip the file locally
            log.info('Opening zip file...')
            zip_ref = zipfile.ZipFile(dirname)
            extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            zip_ref.extractall(extracted_path)
            zip_ref.close()

        elif dirname.endswith('tar.gz'):
            # Logic to unzip tarball locally
            log.info('Opening tarball file...')
            tar_ref = tarfile.open(dirname, 'r:gz')
            extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            tar_ref.extractall(extracted_path)
            tar_ref.close()

        elif dirname.endswith('tar'):
            # Logic to unzip tarball locally
            log.info('Opening tarball file...')
            tar_ref = tarfile.open(dirname, 'r:')
            extracted_path = os.path.join(os.path.dirname(dirname), 'extracted')
            if not os.path.exists(extracted_path):
                os.makedirs(extracted_path)
            tar_ref.extractall(extracted_path)
            tar_ref.close()
        else:
            # Logic for wrong file type
            log.error('Input not a directory')
            raise NotADirectoryError

        imgs = [os.path.join(extracted_path, doc) for doc in os.listdir(extracted_path)]
        for file in imgs:
            results.append(extract_image(file, debug, allow_wildcards))

    log.info('Results extracted sucessfully:')
    log.info(results)

    return results


# mmaser: adding function for reaction scheme extraction
# current on 07/15/2020
# TODO 
def extract_reaction_batch(dirname, out_file=None, debug=False, 
                           allow_wildcards=False, confidence_threshold=None):
    """
    Extracts structures from batch of chemical reaction scheme images.
    
    Required imports:
        chemschematicresolver as csr
        csr.extract_image
        json
        os
        
    ARGUMENTS:
        (str)dirname: directory containing a set of reaction images. should be full path to directory.
        (str)out_file: file name for output json
        (flt)confidence_threshold: tesserocr confidence cuttoff below which label/structure predictions will not be returned. Gets passed to extract_image.
        (Bool)debug: Boolean specifying verbose debug mode.
        (Bool)allow_wildcards: Boolean to indicate whether results containing wildcards are permitted
        
    RETURNS:
        (dict)output_dict: dictionary containing all information about the batch of images
        
    output_dict is saved to json file. currently runs extract_image with 
    count_false_positives=True only.
    """

    try:
        images = os.listdir(dirname)
        log.info(f'Directory opened successfully. {len(images)} files found')
    except:
        log.warning('Could not get files in directory provided. Check that it is the full path of the desired directory')
            
    batch_dict = {}
    for image in images:
        # build image dictionary; so far have:
        # 'num_structures'
        # 'structure_i' with 'label', 'smiles', and center
        
        # check to make sure the file is an image
        if not image.endswith(('.png', '.jpg', '.ppm', '.pdf')):
            continue
        
        image_dict = {}
        # get list of (label, smiles, center) tuples from image as well as
        # the number of false positives (count_false_positives=True returns 
        # both output list and number of false_positives)
        image_output, image_false_positives = extract_image(os.path.join(dirname, image), debug=debug, allow_wildcards=allow_wildcards, confidence_threshold=confidence_threshold, count_false_positives=True)
        
        # fill image dictionary with metadata and extracted structure info
        image_dict['num_structures'] = len(image_output)
        image_dict['false_positives'] = image_false_positives
        
        if len(image_output) > 0:
            # add predictions to image dictionary
            for i, struct in enumerate(image_output):
                struct_dict = {'label': struct[0], 'smiles': struct[1],
                               'center': struct[2]}
                struct_name = f'structure_{i}'
                image_dict[struct_name] = struct_dict
            
        batch_dict[image] = image_dict
    
    # generate json file from image_smiles_dict 
    if not out_file == None:
        if not out_file.endswith('.json'):
            try:
                output = os.path.join(dirname, str(out_file))
            except:
                output = os.path.join(dirname, 'output')
        else:
            try:
                out_name = out_file.split('.')[0]
                output = os.path.join(dirname, out_name)
            except:
                output = os.path.join(dirname, 'output')
                
    else:
        output = os.path.join(dirname, 'output')
    
    # check if out_file already exists and adjust name with counter
    path_count = 1
    path_name = output
    while os.path.exists(f'{output}.json'):
        output = f'{path_name}-{path_count}'
        path_count +=1
        if path_count > 100:
            log.warning('out_file already exists, please choose a new name.')
            break
    
    with open(f'{output}.json', 'w') as data_file:
        json.dump(batch_dict, data_file)
        
    return batch_dict


def get_smiles(diag, smiles, r_smiles, extension='jpg'):
    """ Extracts diagram information.

    :param diag: Input Diagram
    :param smiles: List of smiles from all diagrams up to 'diag'
    :param r_smiles: List of smiles extracted from R-Groups from all diagrams up to 'diag'
    :param extension: Format of image file

    :return smiles: List of smiles from all diagrams up to and including 'diag'
    :return r_smiles: List of smiles extracted from R-Groups from all diagrams up to and including 'diag'
    """

    # Resolve R-groups if detected
    if len(diag.label.r_group) > 0:
        r_smiles_group = get_rgroup_smiles(diag, extension)
        for smile in r_smiles_group:
            label_cand_str = list(set([cand.text for cand in smile[0]]))
            r_smiles.append((label_cand_str, smile[1], diag.center))

    # Resolve diagram normally if no R-groups - should just be one smile
    else:
        smile = read_diagram_pyosra(diag, extension)
        label_raw = diag.label.text
        label_cand_str = list(set([clean_output(cand.text) for cand in label_raw]))
        # mmaser: adding '?' label for samples where no str label is detected
        if len(label_cand_str) == 0:
            label_cand_str = ['?']

        # mmaser: added diag.center to smiles output
        smiles.append((label_cand_str, smile, diag.center))

    return smiles, r_smiles
