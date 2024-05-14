import json
import os
import shutil

data_path = "/data/firebux/datasets-llava/LLaVA1.5/llava_instruct_150k.json"

# GCC_train_000163917.jpg
# 000000163917.jpg

with open(data_path, 'r') as fp:
    data = json.load(fp)

#    os.mkdir('./images')
for k in data:
    # ext = os.path.splitext(k['image'])[1]
    file_name = f'GCC_train_{k["id"][3:]}'

    outputFile = '/data/firebux/datasets-llava/LLaVA1.5/images/%s' % (k['image'])
    shutil.move("/data/firebux/datasets-llava/LLaVA1.5/origin/%s" % file_name, outputFile)


if __name__ == "__main__":
    file = '000000163917.jpg'

    print(f'GCC_train_{file[3:]}')
