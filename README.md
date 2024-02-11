## Fine-tuned Llama-2 for MECPE task

This repository contains the code for the task of Multimodal Emotion-Cause Pair Extraction (MECPE) using fine-tuned Llama-2 13b model. The task is to predict the emotion of each utterance in a given conversation (one of [Ekman's seven emotions](https://www.paulekman.com/universal-emotions/)) while also extracting the causes of that utterance from that conversation. 

#### Usage

It is recommended to start by creating a new virtual environment and installing all the dependencies for the project in it.

For using [virtualenv](https://virtualenv.pypa.io/en/latest/):

```bash
pip install --upgrade virtualenv && \
cd llm-multimodal-ecpe && \
virtualenv env
```

For activating:
```bash
source env/bin/activate
```

For activating on Windows use:
```bash
env\Scripts\activate
```

In case you have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python) installed:

```bash
cd llm-multimodal-ecpe && \
conda create -n env python=3.8 && \
conda activate env
```

Now, install all dependencies using:

```bash
pip install -r requirements.txt
```

### Notebooks:

1. ```emotion_recognition_training.ipynb``` : fine-tunes a ```meta-llama/Llama-2-13b-chat-hf``` LLM to predict the emotion label of a particular utterance in a given conversation. As context, for predicting the emotion label of each utterance, we provide the entire conversation along with the speaker information to guide the prediction.

2. ```emotion_inference.ipynb``` : for performing inference on test data using the fine-tuned Llama-2. The resulting emotion-labelled conversations are stored in the folder results_train or results_test in the file ```emotion_labelled_data.json```

3. ```cause_prediction_training.ipynb``` : fine-tunes another ```meta-llama/Llama-2-13b-chat-hf``` model to predict the cause utterances for a particular utterance in a given conversation. Now, as context, we provide the entire conversation but with all the predicted emotion labels as this adds useful information for guiding cause prediction. We essentially treat this as a two-step process. The model is trained to output a list of cause utterance ids.

4. ```cause_inference.ipynb``` : performs inference on test data using the fine-tuned cause predictor. The final results are stored in the same results_train or results_test folder by the name ```Subtask_2_pred.json```.

### Utils:

```generate_input.py``` : is used for generating the train, test, and validation splits and adds the ```video_name``` to each utterance.

### Implementation Details:

1. Both ```emotion_recognition_training``` and ```cause_prediction_training``` used one **Nvidia A100 40GB** GPU for training. (Available on [Google Colab Pro](https://colab.research.google.com/signup) priced at $11.8/month)

2. We use ```accelerate``` library for offloading the model to CPU and disk. (See: [accelerate](https://huggingface.co/docs/accelerate/en/index))

3. Due to memory constraints, we use Quantized Low-Rank Adaptation for fine-tuning a 4-bit quantized Llama-2 model using ```bitsandbytes``` library. (See : [bitsandbytes](https://github.com/TimDettmers/bitsandbytes))

4. We use ```peft``` library for parameter efficient fine-tuning where we define the configuration for LoRA. (See: [peft](https://huggingface.co/docs/peft/en/index))

5. Supervised fine-tuning is performed using ```trl``` library which provides ```SFTTrainer``` for performing supervised fine-tuning step of RLHF. (See: [trl](https://huggingface.co/docs/trl/en/sft_trainer))

6. Inference is performed using two Tesla T4 16GB GPUs. (Available on [Kaggle](https://www.kaggle.com/) for free (30 hrs/month))

