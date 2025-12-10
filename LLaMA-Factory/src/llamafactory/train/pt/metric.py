from typing import Dict, List, Tuple, Any
import numpy as np
from transformers.trainer_utils import EvalPrediction

class GenAnswerEM:
    def __init__(self, tokenizer, answer_prefix="Answer:"):
        self.tok = tokenizer
        self.answer_prefix = answer_prefix

    def _batch_decode(self, ids) -> List[str]:
        # ids can be list[np.ndarray], np.ndarray, or torch.Tensor
        if isinstance(ids, (list, tuple)):
            # flatten possible list of batches
            arrs = [np.asarray(x) for x in ids]
            ids = np.concatenate(arrs, axis=0)
        if hasattr(ids, "tolist"):
            ids = np.asarray(ids)
        # replace -100 (ignored labels) for decoding
        ids = np.where(ids == -100, self.tok.pad_token_id, ids)
        return self.tok.batch_decode(ids, skip_special_tokens=True)

    def _extract_answer(self, text: str) -> str:
        # naive extractor: take text after the first "Answer:"
        i = text.find(self.answer_prefix)
        return text[i+len(self.answer_prefix):].strip() if i != -1 else text.strip()

    def __call__(self,
                 eval_pred: EvalPrediction,
                 inputs: Any = None,             # provided when include_inputs_for_metrics=True
                 **kwargs) -> Dict[str, float]:
        # 1) decode inputs (original eval text)
        src_texts: List[str] = []
        if inputs is not None:
            # inputs can be dict (concatenated) or list of per-batch dicts
            if isinstance(inputs, list):
                input_ids = np.concatenate([np.asarray(b["input_ids"]) for b in inputs], axis=0)
            else:
                input_ids = np.asarray(inputs["input_ids"])
            src_texts = self.tok.batch_decode(input_ids, skip_special_tokens=True)

        # 2) decode generated predictions
        preds = eval_pred.predictions
        if isinstance(preds, tuple):   # some trainers return (generations, ...)
            preds = preds[0]
        pred_texts = self._batch_decode(preds)

        # 3) decode reference answers from labels
        label_ids = eval_pred.label_ids
        ref_texts = self._batch_decode(label_ids) if label_ids is not None else [""] * len(pred_texts)

        # 4) compute final-answer EM
        assert len(pred_texts) == len(ref_texts)
        em = 0
        for p, r in zip(pred_texts, ref_texts):
            em += int(self._extract_answer(p) == self._extract_answer(r))
        em /= max(1, len(pred_texts))

        # (optional) return a few debug strings for the first samples via Trainer logs
        # But keep metrics dict small; logs can become huge on big eval sets.
        return {"em": float(em), "n_eval": float(len(pred_texts))}
