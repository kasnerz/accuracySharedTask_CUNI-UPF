#!/usr/bin/env python3

from argparse import ArgumentParser
import pandas as pd
import os


HTML_HEADER = """<!DOCTYPE html>
<html>
<head>
<title>Error annotation</title>
<style>
    body {
          max-width: 60em;
          padding-left: 5em;
    }
    .id {
          margin-top: 2em;
          font-weight: bold;
          font-size: 14pt;
    }
    .text {
          line-height: 3em;
    }

    .missed {
          display: inline-block;
          color: #800;
          background: #faa;
          vertical-align: middle;
          line-height: 1em;
    }
    .correct {
          display: inline-block;
          color: #800;
          background: #46db78;
          vertical-align: middle;
          line-height: 1em;
    }
    .incorrect {
          display: inline-block;
          color: #800;
          background: #db9bf2;
          vertical-align: middle;
          line-height: 1em;
    }
    .extra {
          display: inline-block;
          color: #800;
          background: #fa5;
          vertical-align: middle;
          line-height: 1em;
    }
    .type {
          color: white;
          font-size: 8pt;
          display: block;
    }
    .corr {
          color: black;
          font-size: 9pt;
          display: block;
    }
</style>
</head>"""


def load_texts(text_dir):

    res = {}
    for fname in os.listdir(text_dir):
        if not fname.endswith('.txt') or not os.path.isfile(os.path.join(text_dir, fname)):
            continue
        data = ' '.join(open(os.path.join(text_dir, fname), 'r', encoding='UTF-8').readlines()).strip()
        res[fname] = data
    return res


def main(text_dir, gsml_csv, out_csv, games_csv):

    texts = load_texts(text_dir)
    gsml = pd.read_csv(gsml_csv, encoding='UTF-8', na_filter=False)
    out = pd.read_csv(out_csv, encoding='UTF-8', na_filter=False)
    games = pd.read_csv(games_csv, encoding='UTF-8', na_filter=False)

    print(HTML_HEADER)

    for text_id in sorted(texts.keys()):
        text = texts[text_id].split()
        annots = gsml[gsml['TEXT_ID'] == text_id].to_dict('records')
        annots_out = out[out['TEXT_ID'] == text_id].to_dict('records')
        
        game = games[games['TEXT_ID'] == text_id[:-4]].to_dict('records')[0]

        annots_per_token = [[None, None] for _ in range(len(text))]

        for annot in annots:
            start = annot['DOC_TOKEN_START'] - 1
            end = annot['DOC_TOKEN_END'] - 1

            for i in range(start, end+1):
                annots_per_token[i][0] = annot['TYPE']

        for annot in annots_out:
            start = annot['DOC_TOKEN_START'] - 1
            end = annot['DOC_TOKEN_END'] - 1

            for i in range(start, end+1):
                annots_per_token[i][1] = annot['TYPE']

        for i, annot in enumerate(annots_per_token):
            if annot[0] is None and annot[1] is None:
                continue
            elif annot[0] is not None and annot[1] is None:
                text[i] = '<div class="missed">' + '<span class="type">' + "r:" + annot[0] + '</span>' + text[i] + "</div>"
            elif annot[0] is None and annot[1] is not None:
                text[i] = '<div class="extra">' + '<span class="type">' + "h:"  + annot[1] + '</span>' + text[i] + "</div>"
            else:
                if annot[0] != annot[1]: 
                    text[i] = '<div class="incorrect">' + '<span class="type">' + "r:" + annot[0] + " h:" + annot[1] + '</span>' + text[i] + "</div>"
                else:
                    text[i] = '<div class="correct">' + '<span class="type">' + annot[0] + '</span>' + text[i] + "</div>"


        print('\n<div class="id">' + text_id +': <a href="' + game['BREF_BOX'] + '">'
              + game['HOME_NAME'] + ' vs ' + game['VIS_NAME'] + ', ' + game['DATE'] + '</a></div>\n'
              + '<div class="text">\n' + ' '.join(text) + '</div>\n')

    # print footer
    print('\n\n</body>\n</html>')


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('text_dir', type=str, help='Path to texts')
    ap.add_argument('gsml_csv', type=str, help='Path to gsml.csv')
    ap.add_argument('out_csv', type=str, help='Path to the output GSML file')
    ap.add_argument('games_csv', type=str, help='Path to games.csv')

    args = ap.parse_args()
    main(args.text_dir, args.gsml_csv, args.out_csv, args.games_csv)
