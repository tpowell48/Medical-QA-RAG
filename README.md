# Medical-Q-A-RAG
## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to answer medical questions using domain-specific datasets (MedQuAD, MedlinePlus XML) and various open-source LLMs (LLaMA-3, Mistral-7B, Phi-3). It demonstrates how different models perform on medical QA tasks when augmented with external medical corpora.
## Requirements
This notebook was implemented and run originally on Google Colab, packages installations that are not included in the default Colab kernel are included in the notebook.
To run this notebook locally, make sure the following Python packages are installed:
~~~
pip install transformers datasets sentence-transformers faiss-cpu
pip install accelerate evaluate pandas scikit-learn lxml
~~~
You also need access to the HuggingFace Hub, using your account token, for downloading the LLMs and a GPU for efficient inference.
Mistral-7B and Llama-3 are both gated models and require you to request access for use on HuggingFace.

## Data Sources
### MedQuAD CSV
A dataset of medical QA pairs.
Loaded from a CSV and preprocessed for evaluation.

### MedlinePlus XML
Official XML documents containing medical articles.
Parsed using xml.etree.ElementTree to create a corpus for retrieval.

## LLMs Used via HuggingFace Transformers
LLaMA-3 
Mistral-7B
Phi-3

Each model is prompted using both vanilla and retrieval-augmented prompts.

## Workflow Summary
 1. Data Loading:
Load MedQuAD CSV and MedlinePlus XML.

2. Index Construction:
Embed passages using sentence-transformers.
Build a FAISS index for nearest-neighbor retrieval.

3. Model Loading:
Load HuggingFace-compatible LLMs.

4. QA Evaluation:
Compare direct QA vs. RAG-enhanced QA for each model.

5. Scoring:
Evaluate responses using BLEURT and human annotator scores.

## Evaluation
Each response is scored for:
- Factual Accuracy
- Clarity
- Relevance
- Completeness
- Human-likeness (evaluated separately)

You can also manually review the answers using the notebook's interactive scoring cells.

## Usage Instructions
1. Clone or download the notebook.

2. Prepare datasets (medquad.csv, MedlinePlus XML).

3. Run the notebook cell-by-cell, ensuring:
  * GPU is enabled
  * HuggingFace login is complete (for LLaMA-3 or Phi-3)

4. Compare model outputs and analyze the results.

## Results Interpretation
Look at the final DataFrames and visualizations to:
- Identify which model performs best.
- Understand how retrieval augmentation impacts factual accuracy.

## Notes
This notebook is GPU-accelerated and may not run efficiently on CPU.<br>
Consider limiting the number of examples for quick iteration.
