# from lib.models.language_model.bert import BERT
from lib.models.language_model.modeling import BERT
from lib.models.language_model.modeling import TinyBERT

# from lib.checkpoints.language_model.bert_huggingface import BERT_HUGGINGFACE

student_model_config = 'student_model/'

def build_bert():
    # position_embedding = build_position_encoding(cfg)
    train_bert = False
    bert_type = 'pytorch'
    if bert_type == "pytorch":
        bert_model = BERT('bert-base-uncased', '/home/zhanggt/SDTrack/pretrained_models/bert/bert-base-cased.tar.gz', train_bert,256,
                         40, 12)
    else:
        raise ValueError("Undefined BERT TYPE '%s'" % bert_type)
    return bert_model

def build_tiny_bert():
    train_tinybert = True
    continue_train = False
    bert_type = 'pytorch'
    if bert_type == "pytorch":
        tinybert_model = TinyBERT('/home/zhanggt/SDTrack/lib/models/language_model/student_model/',train_tinybert, continue_train)
    else:
        raise ValueError("Undefined BERT TYPE '%s'" % bert_type)
    return tinybert_model