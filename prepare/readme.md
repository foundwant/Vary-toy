

### 数据集准备
- 选取huggingface数据集 BUAADreamer/llava-en-zh-300k
  Vary-toy使用llava-80K的数据集，我们希望识别更细腻精准，所以选取llava-150K的全量数据进行sft
- BUAADreamer/llava-en-zh-300k数据集，包含中文和英文两个数据集描述，有150k个针对图片的描述集合
- 通过llava-en-zh-300k.py脚本下载数据集
- pure_data.py 脚本是LLaVA项目主页推荐的数据集清洗脚本，因数据集描述不全面，不知道和json的映射关系，放弃使用。
