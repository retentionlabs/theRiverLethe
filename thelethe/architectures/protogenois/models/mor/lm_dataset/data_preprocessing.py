class AddLabels:
    def __call__(self, sample: dict):
        attention_mask = sample["attention_mask"]
        labels = sample["input_ids"].clone()
        labels[attention_mask == 0] = -100
        sample.update({"labels": labels})
        return sample


class RemoveIndex:
    def __call__(self, sample: dict):
        sample.pop("index", None)
        return sample