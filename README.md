# Learn Custom LLMs: Tutorial to Develop an LLM for Translating English to Punjabi

Follow the notebook in this repo along to understand the entire process.

In this repo, we'll guide you through the process of fine-tuning a language model to translate English sentences into Punjabi. This project uses a smaller 1 billion parameter model that fits into the free TPU memory of Google Colab. We'll walk you through the steps of using a custom dataset, setting up the training environment, and evaluating the model's performance. By the end, you'll see how this approach can benefit local languages through translation and how you can replicate it at home using open-source tools.

# Introduction to the Project
Language models have become incredibly powerful, but they are often limited to major languages. Our goal is to fine-tune an existing language model to translate English sentences into Punjabi, a regional language. This is particularly important for preserving local languages and making technology accessible to more people. We'll use a simpler 1 billion parameter model from the BLOOM family, which can fit into the free TPU memory of Google Colab, making it easy to experiment with even on a modest setup.

# Preparing the Dataset
The dataset is custom-created, containing pairs of English sentences and their Punjabi translations. Although small, with only 500+ prompts, this dataset serves as a starting point. Each English sentence is followed by its Punjabi translation, formatted to facilitate easy tokenization and training.

# Setting Up the Model and Tokenizer
We begin by loading the BLOOM model and its tokenizer. The tokenizer is responsible for converting the text into a format that the model can understand. Here's how you can initialize them:
from transformers import AutoModelForCausalLM, AutoTokenizer

# Preparing for Fine-Tuning
We load the custom dataset into a DataFrame and convert it into a format suitable for training with the model. Tokenization of the dataset is a key step to ensure the inputs are correctly processed by the model:
import pandas as pd
from datasets import Dataset

# Configuring and Training the Model
We use the PEFT (Parameter-Efficient Fine-Tuning) library to apply LoRA (Low-Rank Adaptation) configurations, which make the training process more efficient. We set up the training arguments and use the Trainer API to handle the training loop:

# Evaluating the Model
After training, we save and reload the fine-tuned model. We then generate translations for new English sentences to evaluate the model's performance:

# Learning and Future Directions
This project demonstrates how to fine-tune a language model for translating English to Punjabi using a small dataset. While the dataset is not diverse enough for production use, it serves as a proof of concept. To improve, one can gather a more extensive and varied dataset, use data augmentation techniques, and experiment with larger models if resources allow. By sharing this approach, we hope to inspire others to explore similar projects and contribute to the preservation and accessibility of local languages.
