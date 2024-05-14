import json
import os
import shutil

data_path = "/data/firebux/datasets-llava/LLaVA1.5/llava_instruct_150k.json"

# GCC_train_000163917.jpg
# 000000163917.jpg

with open(data_path, 'r') as fp:
    data = json.load(fp)

#    os.mkdir('./images')
for k in data.keys():
    ext = os.path.splitext(data[k]['image'])[1]
    file_name = data[k]['id'][3:]

    outputFile = '/data/firebux/datasets-llava/LLaVA1.5/images/%s%s' % (file_name, ext)
    shutil.move("/data/firebux/datasets-llava/LLaVA1.5/", outputFile)


if __name__ == "__main__":
    print("000000163917.jpg"[3:])
