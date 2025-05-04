# Taetz_Bordelius_Continual_ImageCaptioning
Continual learning Image Captioning model with catastrophic forgetting mitigation techniques.

Our model beats previous state of the art results set by ContCap: A Scalable Framework for Continual Image Captioning.

Our best results with continual learning on ContCap continual MS COCO dataset split:
| Metric   | Score  |
|----------|--------|
| BLEU-1   | 66.4   |
| BLEU-2   | 47.9   |
| BLEU-3   | 31.8   |
| BLEU-4   | 21.8   |
| ROUGE-1  | 41.8   |
| ROUGE-2  | 15.1   |
| ROUGE-L  | 37.0   |
| METEOR   | 26.3   |
| CIDEr    | 64.1   |



**Methodology:**

**Multi-Loss Training for Vision-Language Models:**
The proposed method employs a training strategy that combines multiple loss components to enhance the model's performance in generating accurate image captions while improving its ability to recognize objects and distinguish between visual categories. The approach builds upon a standard cross-entropy loss for language modeling but extends it with two additional objectives: a noun-focused loss and a language-guided contrastive loss, dynamically balanced to ensure stable training.


**Base Cross-Entropy Loss:**
The foundation of the training process is the standard cross-entropy loss (loss), computed from the model's language modeling head given the input image and target captions. This loss ensures that the model learns to generate coherent and contextually appropriate descriptions by minimizing the negative log-likelihood of the correct tokens.


**Noun-Based Loss for Object Awareness:**
To reinforce the model's ability to recognize and describe key objects in an image, an supporting loss is introduced. First, nouns are extracted from the ground-truth captions (e.g., "dog," "cat") and reformatted into structured prompts (e.g., "An image of a dog and a cat"). These prompts are then fed back into the model, without gradient computation to compute an additional cross-entropy loss. By explicitly training the model to predict these noun-centric descriptions, the method encourages stronger alignment between visual features and object-related language. The purpose is to evaluate the model's performance on noun-centric prompts without letting this auxiliary loss directly update the model's weights.


**Language-Guided Contrastive Loss for Discriminative Learning:**
For tasks beyond the first task of the training phase, a contrastive loss is applied to improve the model's discriminative capabilities. This component operates in the embedding space:
1.	Text Embedding Extraction: The noun-based prompts are encoded into fixed-dimensional embeddings using a pretrained text encoder.
2.	Similarity Measurement: The cosine similarity is computed between:
The image embedding (anchor) and the correct prompt embedding (positive pair).
The image embedding and embeddings from previous tasks (negative pairs).
3.	Triplet-Loss Optimization: The contrastive objective pushes the model to maximize similarity for correct image-text pairs while minimizing it for incorrect ones, following a margin-based formulation:


**Dynamic Weighting Strategy:**
To prevent any single loss from dominating the optimization, we compute adaptive weights based on their relative magnitudes:
β=Llgcls/Ltotal where Ltotal=LCE(Loss Cross-Entropy) +Llgcls+Lnouns
The final loss combines these components as and is used in backpropagating using gradient scaling.:
L=LCE+β⋅Llgcls+Lnouns


**Model TRAINING diagram**
![diagram](https://github.com/user-attachments/assets/fd6e5fe4-2643-406f-b8cb-e0a9aa3605d0)



**All results**

**ContCap dataset split results:**
| MODEL                                                                                     | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | CIDEr | LGCL           | Time            |
|-------------------------------------------------------------------------------------------|--------|--------|--------|--------|---------|---------|---------|--------|-------|----------------|-----------------|
| NO LGCL                                                                                   | 62.83  | 43.31  | 28.49  | 18.94  | 39.28   | 12.73   | 32.99   | 25.32  | 58.46 | `use_lgcl: False` | 31/03/2025 09:45 |
| SUM+DYNAMIC (best)                                                                        | 66.37  | 47.88  | 31.78  | 21.78  | 41.76   | 15.06   | 36.96   | 26.30  | 64.13 | `use_lgcl: True`  | 02/04/2025 07:25 |
| DYNAMIC                                                                                   | 61.53  | 41.25  | 26.90  | 17.79  | 38.61   | 12.00   | 32.73   | 27.77  | 57.15 | `use_lgcl: True`  | 05/04/2025 07:52 |
| SUM                                                                                       | 62.02  | 42.35  | 27.85  | 18.55  | 39.18   | 12.62   | 32.97   | **28.32**  | 57.02 | `use_lgcl: True`  | 06/04/2025 06:47 |


**Our dataset split results:**
| MODEL                   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | CIDEr | LGCL           | Time            |
|-------------------------|--------|--------|--------|--------|---------|---------|---------|--------|-------|----------------|-----------------|
| No LGCL - Our dataset   | 61.62  | 42.34  | 27.30  | 17.93  | 37.82   | 11.89   | 32.21   | 27.91  | 56.49 | `use_lgcl: False` | 05/04/2025 09:44 |
| LGCL - SUM+DYNAMIC - Our dataset      | 61.00  | 41.97  | 27.38  | 18.00  | 38.24   | 12.37   | 33.29   | 27.73  | 56.89 | `use_lgcl: True`  | 06/04/2025 12:55 |


