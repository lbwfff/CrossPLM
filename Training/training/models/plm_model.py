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
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
    ):
        self.backbone_model_id = backbone_model_id
        self.task_type = task_type
        self.num_labels = num_labels
        self.freeze_backbone = bool(freeze_backbone)
        self.freeze_layers = int(freeze_layers) if freeze_layers is not None else 0

        # trust_remote_code is required for Synthyra/ESM2-* (FastPLMs runtime)
        # but harmless for native facebook/esm2. Always pass True so both
        # model families load without user-side toggling.
        self.config = config or AutoConfig.from_pretrained(
            backbone_model_id, trust_remote_code=True
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            backbone_model_id, trust_remote_code=True
        )
        self.model = model or self._build_model()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        # Apply freeze control after the model is built, before the optimizer is
        # created (Trainer filters trainable parameters).  freeze_backbone takes
        # precedence over freeze_layers.
        if self.freeze_backbone or self.freeze_layers > 0:
            self._apply_freeze()

    def _build_model(self):
        if self.task_type == "mlm":
            return AutoModelForMaskedLM.from_pretrained(
                self.backbone_model_id, trust_remote_code=True
            )
        elif self.task_type == "token_classification":
            if self.num_labels is None:
                raise ValueError("num_labels must be provided for token_classification task")
            self.config.num_labels = self.num_labels
            return AutoModelForTokenClassification.from_pretrained(
                self.backbone_model_id,
                config=self.config,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _apply_freeze(self):
        """Freeze backbone parameters according to freeze_backbone / freeze_layers.

        - freeze_backbone=True: freeze everything except the classification head
          (any parameter whose name contains 'classifier' or 'score' stays trainable).
        - freeze_layers=N>0: freeze the bottom N encoder layers (best-effort
          detection for ESM / BERT-like stacks).
        """
        import re

        if self.freeze_backbone:
            frozen = 0
            trainable = 0
            for name, param in self.model.named_parameters():
                if "classifier" in name or "score" in name or "lm_head" in name:
                    param.requires_grad = True
                    trainable += param.numel()
                else:
                    param.requires_grad = False
                    frozen += param.numel()
            print(f"[Freeze] freeze_backbone=True — frozen {frozen:,} params, trainable {trainable:,} (classifier only)")
            return

        # freeze_layers: try to locate an encoder layer list and freeze bottom N
        n = int(self.freeze_layers)
        if n <= 0:
            return
        # Detect encoder stack for common PLM layouts
        layer_list = None
        # ESM: model.esm.encoder.layer
        try:
            if hasattr(self.model, "esm") and hasattr(self.model.esm, "encoder") and hasattr(self.model.esm.encoder, "layer"):
                layer_list = self.model.esm.encoder.layer
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "encoder") and hasattr(self.model.base_model.encoder, "layer"):
                layer_list = self.model.base_model.encoder.layer
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                layer_list = self.model.encoder.layer
        except Exception:
            layer_list = None

        if layer_list is not None:
            n = min(n, len(layer_list))
            for i in range(n):
                for p in layer_list[i].parameters():
                    p.requires_grad = False
            print(f"[Freeze] freeze_layers={n} — froze bottom {n} encoder layers")
        else:
            # Fallback: name-pattern based freezing for Bert/Roberta/ESM-like
            # encoder.layer.<idx> ordering.
            pattern = re.compile(r"encoder\.layer\.(\d+)")
            frozen_layers = set()
            for name, param in self.model.named_parameters():
                m = pattern.search(name)
                if m and int(m.group(1)) < n:
                    param.requires_grad = False
                    frozen_layers.add(int(m.group(1)))
            if frozen_layers:
                print(f"[Freeze] freeze_layers={n} — froze encoder layers {sorted(frozen_layers)} (pattern match)")
            else:
                print(f"[Freeze] freeze_layers={n} requested but no encoder layer stack detected — no parameters frozen")

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
