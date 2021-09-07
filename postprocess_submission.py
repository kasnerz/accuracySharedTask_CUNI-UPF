#!/usr/bin/env python3
"""
Postprocessing script for merging the subsequent annotated errors with the same type.
This improves precision due to the way errors are "consumed" during evaluation (see README in the shared task repo).
"""
import csv
import sys

fname = sys.argv[1]


def follows(anno_prev, anno):
    """
    Finds if two annotations can be merged.
    """
    text_id1, sentence_id1, annotation_id1, tokens1, sent_token_start1, \
        sent_token_end1, doc_token_start1, doc_token_end1, type1, correction1, comment = anno_prev

    text_id2, sentence_id2, annotation_id2, tokens2, sent_token_start2, \
        sent_token_end2, doc_token_start2, doc_token_end2, type2, correction2, comment = anno

    if text_id1 != text_id2 or sentence_id1 != sentence_id2:
        return False

    return int(sent_token_end1) + 1 == int(sent_token_start2) \
         and int(doc_token_end1) + 1 == int(doc_token_start2) \
         and type1 == type2

def merge_annos(anno_grouped):
    """
    Merges subsequent annotations.
    """
    # use the first anno
    new_anno = anno_grouped[0]
    # merge tokens
    new_anno[3] = " ".join([anno[3] for anno in anno_grouped])
    # set end according to the last anno
    new_anno[5] = anno_grouped[-1][5]
    new_anno[7] = anno_grouped[-1][7]

    return new_anno

rows = []

with open(fname) as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    header = next(reader)
    rows.append(header)

    anno_grouped = []
    
    for anno in reader:
        if not anno_grouped or follows(anno_grouped[-1], anno):
            anno_grouped.append(anno)
        else:
            merged_anno = merge_annos(anno_grouped)
            rows.append(merged_anno)
            anno_grouped.clear()
            anno_grouped.append(anno)

    merged_anno = merge_annos(anno_grouped)
    rows.append(merged_anno)

# overwrites the original file
with open(fname, "w") as f_out:
    writer = csv.writer(f_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for row in rows:
        writer.writerow(row)