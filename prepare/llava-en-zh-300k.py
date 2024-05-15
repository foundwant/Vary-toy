import json

from datasets import load_from_disk, load_dataset


# from datasets import load_from_disk


def download_dataset(dataset_name: str = 'BUAADreamer/llava-en-zh-300k'):
    # 下载的数据集名称,
    # dataset_name = 'keremberke/plane-detection'
    # 数据集保存的路径
    save_path = '/data/firebux/data/'
    # name参数为full或mini，full表示下载全部数据，mini表示下载部分少量数据
    dataset = load_dataset(dataset_name, name='zh')  # or name='en'
    dataset.save_to_disk(save_path)


"""
    llava_instruct_en_zh_300k.json
    {
        "id": "",
        "image": "",
        "conversations": []
    }
"""
target_data = []


def loads_and_conv(path: str):
    """
        加载数据集，存储图片并生成json格式数据
        https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k/viewer?row=0
        BUAADreamer/llava-en-zh-300k: 格式

        ["messages", "images"]
    """
    datasets = load_from_disk(path)
    print(f"len(datasets): {len(datasets)}")

    seq = 1
    print(datasets[0])

    for data in datasets:
        item = {}
        img = data['images']
        img_name = f"{seq}.jpeg"
        img.save(f"/data/firebux/datasets-llava/LLaVA-en-zh-300K/images/{img_name}")

        item['id'] = seq
        item['image'] = img_name
        # messages
        item['conversations'] = []
        for chat in data['messages']:
            item['conversations'].append(
                {
                    "from": chat['role'],
                    "value": chat['content']
                }
            )
        target_data.append(item)

        print(seq)
        seq += 1

        if 100 == seq:
            break

    with open('/data/firebux/datasets-llava/LLaVA-en-zh-300K/llava_instruct_en_zh_300k.json', 'w') as f:
        f.write(json.dumps(target_data, ensure_ascii=False, indent=2, separators=(",", ": ")))

    # datasets.to_list()


if __name__ == "__main__":
    # datasets_name = 'keremberke/plane-detection'
    # download_dataset()
    path = "/data/firebux/data/train"
    loads_and_conv(path)
