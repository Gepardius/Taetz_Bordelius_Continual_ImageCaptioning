
### Base Cross-Entropy Loss (`L_CE`)

The primary supervision signal is provided by the standard cross-entropy loss, computed between the model’s predicted token distributions and the ground-truth captions. This objective ensures the model learns to generate fluent, contextually appropriate captions from input images.

---

### Noun-Based Cross-Entropy Loss (`L_nouns`)

To promote object-awareness, a secondary loss is introduced. Nouns are first extracted from the reference caption (e.g., *"dog"*, *"cat"*) and reformulated into a structured prompt (e.g., *"An image of a dog and a cat"*). These prompts are passed back into the model (without updating its parameters) to compute an additional cross-entropy loss based on the nouns prompt as a target caption. This encourages stronger alignment between visual features and object-specific language representations.

---

### Language-Guided Contrastive Loss (`L_LGCL`)

To further improve discriminative capability, a contrastive loss is applied during multi-task training, inspired by the method proposed by Khan et al. (2023). Image embeddings serve as anchors, while positive and negative text embeddings (derived from current and random previous task prompts, respectively) are used to compute a triplet loss based on cosine similarity. This encourages the model to associate correct image-text pairs while distinguishing them from unrelated prompts.

The LGCL loss is calculated using the triplet loss as follows:

```math
L_{triplet} = 1 - \cos(\mathbf{v}_{img}, \mathbf{v}_{text}^+) + \cos(\mathbf{v}_{img}, \mathbf{v}_{text}^-)
```

Where:
- $\mathbf{v}_{img}$ is the image embedding (anchor),
- $\mathbf{v}_{text}^+$ is the positive text embedding (current caption prompt),
- $\mathbf{v}_{text}^-$ is the negative text embedding (random previous task caption prompt),
- $\cos(\cdot, \cdot)$ denotes cosine similarity.

---

### Dynamic Loss Weighting

To maintain training stability and prevent any single loss from dominating, adaptive weights are computed based on the relative magnitudes of each loss component. Specifically, the weight $eta$ for the contrastive loss is defined as:

```math
eta = rac{L_{LGCL}}{L_{CE} + L_{nouns} + L_{LGCL}}
```

The total loss is formulated as:

```math
L_{total} = L_{CE} + eta \cdot L_{LGCL} + L_{nouns}
```

This combined loss is used for backpropagation.

---

**Note:** Both the noun-focused supervision and the contrastive objectives are employed **only during training**. During inference, the model requires only the input image to generate captions, introducing no additional computational complexity or input requirements.
