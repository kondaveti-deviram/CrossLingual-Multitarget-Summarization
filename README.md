# CrossLingual-Multitarget-Summarization

This project enhances multilingual cross-lingual summarization using a **beam-coherent reranking** method. It leverages transformer-based models like **mT5**, **NLLB**, and **LLMs** to generate summaries in multiple target languages from multilingual document clusters.

## Highlights

- **Multilingual Summarization Backbone**: Fine-tuned `mT5` on the [CrossSum](https://huggingface.co/datasets/csebuetnlp/CrossSum) dataset.
- **Beam-Coherent Reranking**: Uses `Sentence-BERT` to rank beam outputs based on semantic similarity.
- **Pivot-Based Translation**: Uses `mT5` to generate English summaries, then `NLLB` for translation into low-resource languages.
- **LLM-Based Generation**: Uses GPT-4 or Mistral-7B for fluent, instruction-guided summaries.
- **Evaluation**: Uses ROUGE, BLEURT, COMET, BERTScore, and language detection via `fastText`.

## Pipeline Overview

- generate.py # Summary generation pipeline
- dataloading.py # Dataset preparation (CrossSum)
- aggregate.py # Cluster alignment and grouping
- evaluate.py # Evaluation using multiple metrics
- aux_models.py # Beam reranking and pivot translation
- summarizers.py # LLM integration (GPT, Mistral)
- requirements.txt # Python dependencies
- Report.pdf # Project report

## Evaluation Metrics

- ROUGE-2
- BLEURT
- COMET
- BERTScore
- Language Accuracy (via fastText)
