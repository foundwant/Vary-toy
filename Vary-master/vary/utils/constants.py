CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


ROOT_PATH = '/data/public/ucaswei/data/'

CONVERSATION_DATA = {

    # pair 4m
    'laion-coco-4m': {
        'images': '',
        'annotations': '',
    }, 

    'cc665k': {
        'images': "",
        'annotations': "",
    },

    'pdf': {
        'images': "",
        'annotations': "",
    },

    'docvqa_train': {
        'images': "",
        'annotations': "",
    },

    'chartqa_train': {
        'images': "",
        'annotations': "",
    },

    'llava-zh-150k': {
        'images': '/data/firebux/datasets-llava/LLaVA-en-zh-300K/images',
        'annotations': '/data/firebux/datasets-llava/LLaVA-en-zh-300K/llava_instruct_zh_150k.json'
    },

    'llava-en-150k': {
        'images': '/data/firebux/datasets-llava/LLaVA-en-zh-300K/images',
        'annotations': '/data/firebux/datasets-llava/LLaVA-en-zh-300K/llava_instruct_en_150k.json'
    }
}
