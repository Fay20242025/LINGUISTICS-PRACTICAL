from transformers import BertForNextSentencePrediction, Trainer, TrainingArguments
from nltk import wordpunct_tokenize
model_name = "bert-base-multilingual-uncased"
tokenizer = wordpunct_tokenize
model = BertForNextSentencePrediction.from_pretrained(model_name)
def preprocess_data(examples):
    tokenized_examples = tokenizer(examples['sent_a'], examples['sent_b'], truncation=True, padding='max_length', max_length=512)
    return tokenized_examples
data = {'sent_a': ['Lakwet nechesonen mbiret.', 'Awendi sugul karon.'],
        'sent_b': ['Yechesan lakwet ak mbiret komuriti gee.', 'Mara kobwaa toek sugul karon.']}
tokenized_dataset = preprocess_data(data)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()