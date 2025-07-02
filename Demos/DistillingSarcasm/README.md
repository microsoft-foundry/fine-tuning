# Distilling Sarcasm
> or: "how to teach GPT-4.1-nano how to be sarcastic"

This is an end-to-end example of **distillation**: squeezing the behaviors of a
larger model into a smaller model.

## tl;dr

1. Go spin up base model deployments for all models you want to work with. In
   this notebook, I use o3, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano,
   gpt-4o, and gpt-4o-mini. You might need TPM quota in your subscription.

2. Create and populate a `.env` file to simplify stuff. In it, put some Azure
   specific details:

```properties
AZURE_OPENAI_ENDPOINT=https://<YOUR OWN ENDPOING>.openai.azure.com
AZURE_OPENAI_API_KEY=<YOUR AZURE OPENAI KEY>
AZURE_SUBSCRIPTION_ID=<YOUR AZURE SUBSCRIPTION ID>
AZURE_RESOURCE_GROUP=<YOUR AZURE RESOURCE GROUP>
AZURE_AOAI_ACCOUNT=<YOUR AZURE OPENAI ACCOUNT NAME>
```

3. Wrangle the Python stuff. (See below.)

```
$ python3 -m venv .venv
$ . .venv/bin/activate
(venv) $ pip install -r requirements
```

Then launch this sucker in Jupyter notebooks. The easiest way is to fire it up
in **Visual Studio Code**:

```
$ code sarcasm.ipynb
```

Happy distilling. 🧪

## Background
The basis for this demo is the Distillation [demo](https://github.com/azure-ai-foundry/build-2025-demos/blob/main/Azure%20AI%20Model%20Customization/DistillationDemo/demo.ipynb)
featured at Build 2025:

And the [tutorial](https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune)
on fine-tuning gpt-4o with the prompt:
_Clippy is a factual chatbot that is also sarcastic._

## Grader
We use a crude grader that rewards two things:

1. The amount of sarcasm from 1 (no sarcasm) to 10 (the most sarcasm)
2. Correct answers to the users questions.

More importantly, it *heavily* penalizes wrong answers by driving the scores to
zero. In practice, I don't think I saw this happen, but it's a fun example.

We leave it to **o3** to decide what we mean by sarcasm 😜

## Methodology
More will be written here, but at a high-level the approach takes a few steps.

For now, just read the [notebook](./sarcasm.ipynb) 😉

### 0. Assemble Human Curated Data
We need something of a "gold standard." In this case, we use some pre-canned
examples from the Azure OpenAI docs.

### 1. Build & Test our Grader
We assemble a Grader: a combination of model (o3) and a prompt. The prompt
tells the grader how to score other models' output.

### 2. Benchmark our Base Models
We use the "gold standard" to check if our Grader is doing its job. If we can't
trust the grader, who can we trust?

### 3. Pick our Teacher and our Student
We then give an assignment to our base models (`o3`, `o4-mini`, `4.1-*`, 
`4o-*`) and have the Grader decide on their scores.

We use these scores to determine which base model shows the most aptitude
for our use case. That model we pick as our Teacher.

We also figure out who our Student should be based on which model performs the
worst...yet might be much cheaper to use than our Teacher. (We're on a fixed
budget!)

### 4. Distill from the Teacher
We take the Teacher's answers to the questions and consider them what the
Student needs to learn. We turn them into our training data for fine-tuning.

### 5. Train our Student
The Student gets to work learning by studying the Teacher's output.

### 6. Test our Student against its Peer
Now, to keep things fair, we take _new_ data and ask the Student to generate
responses. We also ask its peer, the un-trained version of itself, to do the
same. We then compare the two.

### 7. Celebrate or Cry
If our Student bests the Peer, we celebrate! Our job is done.

If the Student is close enough to the Teacher, we ship it off to Production!
