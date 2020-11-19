import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, NewType, Tuple, Union
import torch
import warnings
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
import linecache
from filelock import FileLock
from transformers import pipeline, PreTrainedTokenizer, BertTokenizerFast, TrainingArguments
from transformers import AlbertConfig, AlbertModel, AlbertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer

def get_amino_acid_weights_by_frequency(tokenizer):
    
    """
    Amino acid frequency in all Pfam sequences.
        It is used for masked_lm_weights. (inversely weighted; frequent AA -> low weight)
        This feature for triplets is not available yet.
        Some amino acids are too rare to be considered, so manually set to a small value
        Manual values (small): set to large values so their weights are low 
            B=0.1500
            J=0.8000
            X=0.8000
            Z=0.1500
        Manual values (large): set to small values so their weights are high
            O=0.0150
            U=0.0150
    """
    AA_FREQ = {
        "a": 0.0900, "b": 0.1500, "c": 0.0142, "d": 0.0548, 
        "e": 0.0604, "f": 0.0419, "g": 0.0760, "h": 0.0230, 
        "i": 0.0604, "j": 0.8000, "k": 0.0490, "l": 0.1014, 
        "m": 0.0228, "n": 0.0372, "o": 0.0150, "p": 0.0439, 
        "q": 0.0353, "r": 0.0557, "s": 0.0616, "t": 0.0541, 
        "u": 0.0150, "v": 0.0739, "w": 0.0132, "x": 0.8000, 
        "y": 0.0313, "z": 0.1500
    }
    class_weights = [0.0]*tokenizer.vocab_size
    AA_WEIGHT = {a:0.05/AA_FREQ[a] for a in AA_FREQ.keys()}
    for aa in AA_WEIGHT.keys():
        idx=tokenizer.get_vocab()[aa]
        class_weights[idx]=AA_WEIGHT[aa]
    return torch.tensor(class_weights, dtype=torch.float)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser=argparse.ArgumentParser(description="Pretrain ALBERT-seq")
    parser.add_argument('--train_data',type=str,required=True)
    parser.add_argument('--eval_data',type=str,required=True)
    parser.add_argument('--output_dir',type=str,required=True)
    parser.add_argument('--train_batch_size',type=int,default=256)
    parser.add_argument('--eval_batch_size',type=int,default=256)
    parser.add_argument('--learning_rate',type=float,default=0.00176)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--num_warmup_steps',type=int,default=5000)
    parser.add_argument('--logging_steps',type=int,default=5000)
    parser.add_argument('--save_steps',type=int,default=5000)
    parser.add_argument('--save_total_limit',type=int,default=20)
    parser.add_argument('--train_input_mode',type=str,default='lazy')
    parser.add_argument('--eval_input_mode',type=str,default='fast')
    parser.add_argument('--weighted_loss',type=str2bool,nargs='?',const=True,
                       default=False, help='if true, amino acid frequencies are used to compute weighted loss')
    parser.add_argument('--vocab_file',type=str,default='./pfam_vocab.txt')
    parser.add_argument('--model_max_length',type=int,default=384,help='maximum length of single sequence')
    parser.add_argument('--masked_lm_prob',type=float,default=0.15,help='Probability of tokens to be masked')
    parser.add_argument('--disable_tqdm', type=str2bool, nargs='?',const=True, 
                        default=False, help='Disables tqdm for progress log')
    return parser.parse_args()

def _collate_batch(examples, tokenizer):
    """Custom function. Expects tensorized examples in dict with key=input_ids"""
    examples = [e["input_ids"] for e in examples]
    return torch.tensor(examples, dtype=torch.long)
    #return torch.stack(torch.tensor(examples, dtype=torch.long), dim=0)

@dataclass
class DataCollatorForLanguageModeling_custom(DataCollatorForLanguageModeling):

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
#         if isinstance(examples[0], (dict, BatchEncoding)):
#             batch = self.tokenizer.pad(examples, return_tensors="pt")
#         else:
#             batch = {"input_ids": _collate_batch(examples, self.tokenizer)}
        batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask)
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 90% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.9)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels
    
class AlbertForMaskedLM_weighted(AlbertForMaskedLM):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class LineByLineTextDatasetWithPadding(Dataset):
    """
    A class (LineByLineTextDataset) from transformers v3.5. Modified to pad sequences to a fixed length
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str):

        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        _sequence_length_maximum = 2048 #if model sequence length is longer than this, it is truncated
        if tokenizer.model_max_length > _sequence_length_maximum:
            print("Maximum sequence length is too long ({}). Adjusted to {}.".format(
                tokenizer.model_max_length, _sequence_length_maximum))
            tokenizer.model_max_length = _sequence_length_maximum
        batch_encoding = tokenizer.__call__(lines,
                                            padding='max_length', 
                                            max_length=tokenizer.model_max_length,
                                            truncation=True,
                                            return_attention_mask=False,
                                            return_token_type_ids=False
                                           )
        self.examples = [{"input_ids": e} for e in batch_encoding["input_ids"]]
        
        #self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in batch_encoding["input_ids"]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

class LazyTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, filepath: str):
        # get absolute path
        # convert to str, linecache doesn't accept Path objects
        self.filepath = os.path.abspath(filepath)
        self.num_samples = self._get_n_lines(self.filepath)
        self.tokenizer = tokenizer

    @staticmethod
    def _get_n_lines(filepath):
        with open(filepath, encoding='utf-8') as fhin:
            for line_idx, _ in enumerate(fhin, 1):
                pass

        return line_idx

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        # linecache starts counting from one, not zero, +1 the given index
        seq = linecache.getline(self.filepath, idx+1).strip()
        enc = self.tokenizer.__call__(seq,
                                 padding='max_length',
                                 max_length=self.tokenizer.model_max_length,
                                 truncation=True,
                                 return_attention_mask=False,
                                 return_token_type_ids=False
                                )
        return {"input_ids": enc["input_ids"]}
        #return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long)}

    def __len__(self):
        return self.num_samples

def compute_metrics(pred):
    #expected to have a negative value (e.g. -100) for positions to be ignored
    nz_rows, nz_cols = np.where(pred.label_ids > 0)
    labels = pred.label_ids[nz_rows,nz_cols]
    preds = pred.predictions.argmax(-1)[nz_rows,nz_cols]

    acc = float(np.sum(labels==preds))/float(nz_rows.shape[0])
    return {
        'accuracy': acc,
    }
    
def _load_dataset(file_path,tokenizer,mode='lazy',timer=True):
    #prefix _ to avoid collision with method in datasets module
    since = time.time()
    if mode.lower() == 'lazy':
        dataset = LazyTextDataset(tokenizer=tokenizer,filepath=file_path)
    elif mode.lower() in ['fast', 'single']:
        dataset = LineByLineTextDatasetWithPadding(tokenizer=tokenizer,file_path=file_path)
#    elif mode.lower() in ['split', 'multiple']:
#        from datasets import load_dataset
#        data_files = [os.path.join(os.path.abspath(file_path), f) for f in os.listdir(file_path)]
#        dataset = load_dataset('text', data_files=data_files, split='train', 
#                               cache_dir=os.path.join(os.path.abspath(file_path),'datasets'))
#        def encode(examples):
#            return tokenizer(examples['text'],
#                             padding='max_length',
#                             truncation=True,
#                             return_attention_mask=False,
#                             return_token_type_ids=False
#                    )
#        dataset = dataset.map(encode, batched=True)
#        dataset.set_format(type='torch',columns=["input_ids"])
    else:
        raise ValueError("DataLoader Mode {} not supported".format(mode))
    if timer:
        elapsed = time.time()-since
        print("{:.2f} seconds for training dataset prep (mode {})".format(elapsed,mode))
    return dataset

def main(args):

    tokenizer = BertTokenizerFast(args.vocab_file,do_lower_case=True,truncation=True)
    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = 'right'
    train_dataset = _load_dataset(file_path=args.train_data,tokenizer=tokenizer,mode=args.train_input_mode,timer=True)
    eval_dataset = _load_dataset(file_path=args.eval_data,tokenizer=tokenizer,mode=args.eval_input_mode,timer=True)
    
    albert_config =   {
      "attention_probs_dropout_prob": 0.0,
      "hidden_act": "gelu_new",
      "hidden_dropout_prob": 0.0,
      "embedding_size": 128,
      "hidden_size": 312,
      "initializer_range": 0.02,
      "intermediate_size": 1248,
      "max_position_embeddings": tokenizer.model_max_length,
      "num_attention_heads": 12,
      "num_hidden_layers": 4,
      "num_hidden_groups": 1,
      "layer_norm_eps": 1e-12,
      "inner_group_num": 1,
      "type_vocab_size": 2,
      "vocab_size": tokenizer.vocab_size
    }
    albertconfig = AlbertConfig().from_dict(albert_config)

    if args.weighted_loss:
        print("Frequency-based amino acid class weights are used to compute weighted cross entropy.")
        class_weights = get_amino_acid_weights_by_frequency(tokenizer)
    else:
        class_weights = None
    model = AlbertForMaskedLM_weighted(albertconfig,class_weights = class_weights)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=args.logging_steps,
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        do_eval=True,
        metric_for_best_model="accuracy",
        warmup_steps=args.num_warmup_steps,
        weight_decay=0.01,
        disable_tqdm=args.disable_tqdm,
        logging_dir=args.output_dir
    )

    
    data_collator = DataCollatorForLanguageModeling_custom(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.masked_lm_prob
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #callbacks=[CustomPrinterCallback],
        compute_metrics=compute_metrics
    )
    
    print("Start training")
    trainer.train()
    print("Start evaluation")
    trainer.evaluate()

if __name__ == '__main__':
    args=parse_args()
    main(args)
