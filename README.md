# Assignment: Building Your Own Transformer

## Objective 
- Build Transformer from scratch for German-to-English translation based on 'Attention Is All You Need(Vaswani et al., NeurIPS 2017)'

## Resources
- Dataset: wmt/wmt19 (Datasets at Hugging Face)
- Tokenizer: bert-base-uncased
- Training & Evaluation
  - Loss function: Cross-entropy loss
  - Optimizer: Adam
  - Eval metrics: ppl, BLEU

## Results 
### Epoch 1
- Loss: 3.5671 
- PPL: 35.4146 
- BLEU: 11.8181 | 1-gram: 38.4997 | 2-gram: 15.9742 | 3-gram: 7.8286 | 4-gram: 4.0516

### Epoch 2, global step 48,000 (in progress)
- Loss: 3.3419
- PPL: 28.2736
- BLEU: 16.9334 | 1-gram: 49.4387 | 2-gram: 22.5314 | 3-gram: 11.6987 | 4-gram: 6.3093

## Environments
- Install the required Python packages
    ```bash
    pip install -r requirements.txt
    ```

