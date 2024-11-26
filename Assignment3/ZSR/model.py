import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

class TestTimeTrainingModel(torch.nn.Module):
    def __init__(self, bert_model, tokenizer):
        super().__init__()
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("The tokenizer does not have a [MASK] token.")

    def forward(self, inputs):
        outputs = self.bert_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        cls_embeddings = hidden_states[-1][:, 0, :]
        return cls_embeddings

    def test_time_training(self, inputs, steps=1, mask_ratio=0.15):
        orig_weights = {name: param.clone() for name, param in self.bert_model.named_parameters()}
        for _ in range(steps):
            masked_inputs = self._mask_inputs(inputs, mask_ratio)
            outputs = self.bert_model(**masked_inputs)  # Forward pass
            loss = outputs.loss  # Retrieve the loss
            loss.backward()  # Backward pass
            for param in self.bert_model.parameters():
                param.data -= 0.001 * param.grad
        embeddings = self.forward(inputs)  # Compute embeddings
        self._restore_model(orig_weights)
        return embeddings

    def _mask_inputs(self, inputs, mask_ratio):
        mask = torch.rand(inputs["input_ids"].shape, device=inputs["input_ids"].device) < mask_ratio
        masked_inputs = inputs.copy()  # Avoid modifying the original inputs
        masked_inputs["input_ids"] = inputs["input_ids"].clone()
        masked_inputs["labels"] = inputs["input_ids"].clone()
        masked_inputs["input_ids"][mask] = self.mask_token_id
        masked_inputs["labels"][~mask] = -100
        return masked_inputs

    def _restore_model(self, orig_weights):
        for name, param in self.bert_model.named_parameters():
            param.data = orig_weights[name]

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name)

    test_model = TestTimeTrainingModel(bert_model, tokenizer)

    inputs = tokenizer("Example input sentence.", return_tensors="pt")
    embeddings = test_model.test_time_training(inputs, steps=1, mask_ratio=0.15)
    print(embeddings)
