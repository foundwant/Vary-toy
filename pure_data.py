import json
import os
import shutil

data_path = "/data/firebux/datasets-llava/LLaVA1.5/llava_instruct_150k.json"

# GCC_train_000163917.jpg
# 000000163917.jpg

with open(data_path, 'r') as fp:
    data = json.load(fp)

count = 1
target_data = []
#    os.mkdir('./images')
for k in data:
    if count >= 10000:
        break

    ext = os.path.splitext(k['image'])[1]
    origin_name = f'GCC_train_{k["id"][3:]}'
    output_name = k['image']

    origin_file = "/data/firebux/datasets-llava/LLaVA1.5/origin/%s%s" % (origin_name, ext)
    output_file = '/data/firebux/datasets-llava/LLaVA1.5/images/%s' % output_name

    if os.path.exists(origin_file):
        target_data.append(k)
        shutil.move(origin_file, output_file)
        count += 1
    else:
        print(f"{origin_file} not exist.")


if __name__ == "__main__":
    file = '000000163917.jpg'

    print(f'GCC_train_{file[3:]}')
