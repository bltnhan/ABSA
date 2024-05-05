# from underthesea import sent_tokenize, word_tokenize
# import torch
# from transformers import AutoModelForTokenClassification, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
# phobert_ner = AutoModelForTokenClassification.from_pretrained('/opt/github/Phobert-Named-Entity-Reconigtion/phobert-ner-mck')
#
# label_list = [
#     "B-CNAME",
#     "B-HNAME",
#     "I-CNAME",
#     "I-HNAME",
#     "MCK",
#     "O"
# ]
# content = "Công ty Cổ phần Dầu Thực vật Tường An (mã CK: TAC) đã công bố BCTC quý 1/2022. \
# Cụ thể, doanh thu thuần đạt 1.697 tỷ đồng tăng 7,2% so với cùng kỳ. Tuy nhiên giá vốn hàng bán chiếm tới 94% trong doanh thu khiến lãi gộp chỉ còn hơn 98 tỷ đồng, giảm 55% so với quý 1/2021. Trong kỳ chi phí bán hàng giảm mạnh từ 113,5 tỷ đồng xuống còn hơn 21 tỷ đồng, chi phí QLDN cũng thấp hơn cùng kỳ. Do lãi gộp thấp nên kết quả TAC vẫn báo lãi sau thuế giảm 32% so với cùng kỳ, đạt 53 tỷ đồng – tương đương EPS đạt 1.559 đồng."
# paragraphs = [p for p in content.split("\n") if p != '']
#
# tokenize_sentences = []
# for paragraph in paragraphs:
#     # split sentence with underthesea sent_tokenize
#     sentences = sent_tokenize(paragraph)
#     for sentence in sentences:
#         if sentence != '':
#             # tokenize the sentence
#             tokenize_sentence = word_tokenize(sentence, format="text")
#             tokenize_sentences.append(tokenize_sentence)
#
# for sentence in tokenize_sentences:
#     list_ids = tokenizer(sentence)['input_ids']
#     # if sentence is longer than 256 tokens, ignore token 257 onward
#     # if len(list_ids) >= 256:
#     # list_ids = list_ids[0:255]
#
#     # lấy id của các tokens tương ứng
#     input_ids = torch.tensor([list_ids])
#
#     # không dùng tokenize(decode(encode)), text sẽ bị lỗi khi tokenize do conflict với tokenizer mặc định
#     # lấy các token để đánh tags
#     tokens = tokenizer.convert_ids_to_tokens(list_ids)
#
#     outputs = phobert_ner(input_ids).logits
#
#     predictions = torch.argmax(outputs, dim=2)
#
#     for i in [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())]:
#         print(i)
#         if i[1] != 'O':
#             print(f'tagged tokens: {i}')
#     print('---------------------------------------------')

# import wget
# # url = r'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar'
#
# # url = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab'
# url = 'https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr'
# filename = wget.download(url)
# print(filename)
# See more details at: https://github.com/vncorenlp/VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP(r"C:\Users\Admins\PycharmProjects\pytorch\transformers\vncorenlp\VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
#
# # Input
# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
#
# # To perform word (and sentence) segmentation
# sentences = rdrsegmenter.tokenize(text)
# for sentence in sentences:
# 	print(" ".join(sentence))

import logging
import os
from importlib import import_module
from typing import Dict, List, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer
)
from train.utils_ner import Split, TokenClassificationDataset, TokenClassificationTask

logger = logging.getLogger(__name__)


# thông số cho chạy script
# --------------------------
# vị trí file validation/test
data_dir = 'H:/Work from home/better-kw-15455-data'
# task NER của Transformers
task_type = 'NER'
# vị trí model bert
model_name_or_path = 'H:/Work from home/real_estate_ner_kw_15455'
# vị trí file labels (O, CDT, DA)
labels_path = 'H:/Work from home/labels.txt'
# vị trí lưu kết quả prediction (f1_score và predictions)
output_dir = 'H:/Work from home'


def main():

    # -----------------------------------------------
    # gọi token_classfication_task
    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )
    # -----------------------------------------------

    # -----------------------------------------------
    # tạo label cho prediction
    labels = token_classification_task.get_labels(labels_path)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    # -----------------------------------------------

    # -----------------------------------------------
    # config cho models (tạo tokenizer, labels, ...)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    # -----------------------------------------------

    # -----------------------------------------------
    # trả ra predictions labels theo dạng list tương ứng với word
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list
    # -----------------------------------------------

    # -----------------------------------------------
    # tính f1-score
    # TODO: thêm f1 riêng từng tag
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
    # -----------------------------------------------

    # Initialize our Trainer
    # hàm trainer của transformers, cơ bản giúp cho việc xử lý model (mới được hugging face thêm vào)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    # ------------------------------------------------------------

    # -----------------------------------------------
    # khởi tạo dữ liệu cho predictions
    test_dataset = TokenClassificationDataset(
        token_classification_task=token_classification_task,
        data_dir=data_dir,  # location to folder with file name test.txt
        tokenizer=tokenizer,
        labels=labels,
        model_type=config.model_type,
        mode=Split.test,
    )

    predictions, label_ids, metrics = trainer.predict(test_dataset)
    preds_list, _ = align_predictions(predictions, label_ids)
    # -----------------------------------------------

    # Save f1_score
    output_test_results_file = os.path.join(output_dir, "test_results.txt")
    if trainer.is_world_process_zero():
        with open(output_test_results_file, "w") as writer:
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Save predictions
    output_test_predictions_file = os.path.join(output_dir, "test_predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_dir, "test.txt"), "r") as f:
                token_classification_task.write_predictions_to_file(writer, f, preds_list)


if __name__ == "__main__":
    main()