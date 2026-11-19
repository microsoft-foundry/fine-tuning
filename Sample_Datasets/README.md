# Sample Datasets for Fine Tuning

We've prepared sample datasets to help you understand how to format data for fine-tuning and test end-to-end workloads. These datasets are intended as examples; they are not intended to produce production quality models.

## Dataset integrity before upload

Preserving source bytes is not evidence that labels or generated code are valid.
The [MedMCQ notice](Reinforcement_Fine_Tuning/MedMCQ/README.md) blocks its unlabelled
validation file from grading. The [NL-to-Python notice](Supervised_Fine_Tuning/Text-NL-to-Python/README.md)
documents syntax-audit/explicit-curation requirements and the original asset
hashes. [dataset_integrity.py](dataset_integrity.py) provides offline checks and
source-traceable preparation; it never calls a model, executes generated code,
downloads data or uploads files. Prepared artifacts are not bundled by this
change. [validation-status.json](validation-status.json) separates source metadata
and review-reported counts from checks still requiring parent execution.

## Data Format Documentation

Each technique has specific data format requirements. See the schema documentation for details:

| Technique | Schema Documentation |
|-----------|---------------------|
| **SFT** | [Supervised_Fine_Tuning/SCHEMA.md](./Supervised_Fine_Tuning/SCHEMA.md) |
| **DPO** | [Direct_Preference_Optimization/SCHEMA.md](./Direct_Preference_Optimization/SCHEMA.md) |
| **RFT** | [Reinforcement_Fine_Tuning/SCHEMA.md](./Reinforcement_Fine_Tuning/SCHEMA.md) |

## Training Techniques
We offer three training techniques to optimize your models:

- **Supervised Fine-Tuning (SFT):** Foundational technique that trains your model on input-output pairs, teaching it to produce desired responses for specific inputs.
    - *Best for:* Most use cases including domain specialization, task performance, style and tone, following instructions, and language adaptation.
    - *When to use:* Start here for most projects. SFT addresses the broadest number of fine-tuning scenarios and provides reliable results with clear input-output training data.
    - *Supported Models:* GPT 4o, 4o-mini, 4.1, 4.1-mini, 4.1-nano; Llama 2 and Llama 3.1; Phi 4, Phi-4-mini-instruct; Mistral Nemo, Ministral-3B, Mistral Large (2411); NTT Tsuzumi-7b
    - *Sample Datasets*: 
         - [Vision Fine Tuning](./Supervised_Fine_Tuning/Multimodal-chartqa/) - Fine Tuning for Chart Interpretation
        - [Text Fine Tuning](./Supervised_Fine_Tuning/Text-GSM8K/) - Grade School Math dataset
        - [Fine Tuning with Function Calls](./Supervised_Fine_Tuning/Tool-Calling) - Stock market tool calling example
        - [NL-to-Python Distillation](./Supervised_Fine_Tuning/Text-NL-to-Python/) - Synthetic code generation data via NVIDIA Data Designer
        - [Bug Detection](./Supervised_Fine_Tuning/Text-Bug-Detection/) - Code bug identification, explanation, and fix suggestions

- **Direct Preference Optimization (DPO):** Trains models to prefer certain types of responses over others by learning from comparative feedback, without requiring a separate reward model.
    - *Best for:* Improving response quality, safety, and alignment with human preferences.
    - *When to use:* When you have examples of preferred vs. non-preferred outputs, or when you need to optimize for subjective qualities like helpfulness, harmlessness, or style. Use cases include adapting models to a specific style and tone, or adapting a model to cultural preferences.
    - *Supported Models:* GPT 4o, 4.1, 4.1-mini, 4.1-nano
    - *Sample Datasets*: 
         -  [Orca DPO samples](./Direct_Preference_Optimization/orca_dpo_pairs/) - Preference data with pinned Microsoft/Intel references, license scope, and derivation limitations
       
- **Reinforcement Fine-Tuning (RFT):** Uses reinforcement learning to optimize models based on reward signals, allowing for more complex optimization objectives.

    - *Best for:* Complex optimization scenarios where simple input-output pairs aren't sufficient.
    - *When to use:* RFT is ideal for objective domains like mathematics, chemistry, and physics where there are clear right and wrong answers and the model already shows some competency. It works best when lucky guessing is difficult and expert evaluators would consistently agree on an unambiguous, correct answer. Requires more ML expertise to implement effectively.
    - *Supported Models:* o4-mini
    - *Sample Datasets*: 
            - [Clause Matching](./Reinforcement_Fine_Tuning/ClauseMatching/) - legal contract dataset
        -  [Med MCQ](./Reinforcement_Fine_Tuning/MedMCQ/) - Multiple Choice Medical Q&A

**Most customers should start with SFT,** as it addresses the broadest number of fine-tuning use cases.