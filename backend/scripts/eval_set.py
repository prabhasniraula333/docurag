"""Evaluation questions, expected pages, and ground-truth substrings for the Transformer paper."""

EVAL_SET = [
    {
        "question": "How many GPUs were used to train the models and for how long?",
        "expected_pages": [2, 7, 8],
        "answer_substring": "P100",
    },
    {
        "question": "How is scaled dot-product attention defined?",
        "expected_pages": [3],
        "answer_substring": "weighted sum of the values",
    },
    {
        "question": "What BLEU score did the big model achieve on English-to-German translation?",
        "expected_pages": [1, 8],
        "answer_substring": "28.4",
    },
    {
        "question": "How many FLOPs did training the big model take?",
        "expected_pages": [8],
        "answer_substring": "28.4 41.0 2.3",
    },
    {
        "question": "What BLEU score did the big model get on English-to-French translation?",
        "expected_pages": [1, 8],
        "answer_substring": "41.0",
    },
    {
        "question": "How many layers are in the encoder and decoder stacks?",
        "expected_pages": [2, 3],
        "answer_substring": "6 identical layers",
    },
    {
        "question": "How many parallel attention heads does the base Transformer use?",
        "expected_pages": [5],
        "answer_substring": "8 parallel attention",
    },
    {
        "question": "What is the dimensionality of the model's embeddings and sub-layer outputs?",
        "expected_pages": [3, 5],
        "answer_substring": "dmodel = 512",
    },
    {
        "question": "Which optimizer was used during training?",
        "expected_pages": [7],
        "answer_substring": "adam optimizer",
    },
    {
        "question": "How many warmup steps were used for the learning rate schedule?",
        "expected_pages": [7],
        "answer_substring": "4000",
    },
    {
        "question": "What dropout rate was applied during training of the base model?",
        "expected_pages": [7, 8],
        "answer_substring": "pdrop = 0.1",
    },
    {
        "question": "What beam size and length penalty were used during inference?",
        "expected_pages": [8],
        "answer_substring": "beam size of 4",
    },
    {
        "question": "What hardware did the authors train on?",
        "expected_pages": [7],
        "answer_substring": "8 nvidia p100",
    },
    {
        "question": "Why do they divide the dot products by a scaling factor?",
        "expected_pages": [4],
        "answer_substring": "extremely small gradients",
    },
    {
        "question": "How long did it take to train the smaller base model?",
        "expected_pages": [7],
        "answer_substring": "12 hours",
    },
]

OFF_TOPIC = [
    "What is the capital of France?",
    "What is the best way to bake bread?",
    "Who won the football world cup in 2018?",
]
