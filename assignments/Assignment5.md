# Teaching Weird AI to Write Lyrics

## Overview

By the end of this assignment you should be able to:
- Generate text using Weird AI
- Calculate cross-entropy loss
- Measure training and validation performance
- Implement a training loop
- Observe loss reduction during training
- Save model checkpoints

## Part 1: Notebook

Open the jupyter notebook titled ```05_pretraining.ipynb```

Read the markdown explanation for each section and add the code implementation described to the notebook.

>  Note: Run the notebook cells in order.  If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application implementation

Read the application code and comments in the following files:
- losses.py
- generation.py
- trainer.py

Complete the TODO tasks in each.

Make sure all of the files pass the tests in *test_losses.py* and *test_trainer.py*.  There are tests for each of the files you will work on in this lesson.

You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/<testfile>.py
```

When all tests are passing be sure to check your code into your GitHub repository.

## Part 3: Reflection Questions
1. Why does an untrained model generate nonsense text?
The model has not learned how to pick the next token, so it is just picking a random token. Thats where the loss of 11 comes from, a 1/50000 chance that the model picks the "correct" next token.
2. What is the difference between logits and probabilities?
Logits are the raw score, where the probability is after you take the logits and pass them through softmax.
3. Why is cross-entropy loss useful?
Cross entropy allows us to understand when the model is confidently incorrect. By using a logarithmic scale we are able to adjust based on more than just correctness,
we can adjust based on confidence as well.
4. What does perplexity measure?
This measures how surprised the model is when predicting the next token. It is a comparison between the tokens the model was likely to have chosen
and the expected next token.
5. Why do we need both training and validation datasets?
Otherwise the model would always produce exact matches to training sets.
6. What role does backpropagation play in learning?
This is the mechanism that changes weights based on whether or not the output is what was expected or not.
7. Why should model checkpoints be saved during training?
This allows recovery if there is hardware failure etc. Additionally it allows us to revert to older checkpoints if the model begins to overfit.
Submit your document and a zip file of your current project as two separate attachments.
