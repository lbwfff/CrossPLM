from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


class PLMModel:
    def __init__(
        self,
        backbone_model_id: str,
        task_type: str = "token_classification",
        num_labels: int = None,
        tokenizer=None,
        model=None,
        config=None,
    ):
        self.backbone_model_id = backbone_model_id
        self.task_type = task_type
        self.num_labels = num_labels

        self.config = config or AutoConfig.from_pretrained(backbone_model_id)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(backbone_model_id)
        self.model = model or self._build_model()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

    def _build_model(self):
        if self.task_type == "mlm":
            return AutoModelForMaskedLM.from_pretrained(self.backbone_model_id)
        elif self.task_type == "token_classification":
            if self.num_labels is None:
                raise ValueError("num_labels must be provided for token_classification task")
            self.config.num_labels = self.num_labels
            return AutoModelForTokenClassification.from_pretrained(
                self.backbone_model_id,
                config=self.config,
            )
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
