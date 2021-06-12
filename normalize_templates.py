#!/usr/bin/env python3

import re
import os
import glob

template_path = "generated/simple_templates"
out_dir = template_path + "/normalized"

os.makedirs(out_dir, exist_ok=True)
file_list = glob.glob(os.path.join(template_path, '*.txt'))
out_lines = []

for file in file_list:
    out_fname = os.path.basename(file)

    with open(file) as f, open(out_dir + "/" + out_fname, "w") as f_out:
        for line in f.readlines():
            line = re.sub(r"(\d+)-(\d+)", r"\1 - \2", line)
            line = re.sub(r"(\d+)%", r"\1 percent", line)
            line = re.sub(r"(\s)\s+", r"\1", line)

            f_out.write(line)