# Efficient-Multi-user-Offloading-of-Personalized-Diffusion-Models

## About the Training

This project utilizes Hugging Face's Dreambooth technology for training both cluster wide models and local personalized models. For further details on Dreambooth, refer to the [Hugging Face Dreambooth webpage]([https://huggingface.co/docs](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)).

### Data Sets

- The fine-tuning dataset for the **cluster wide model** is located in **colloection**.
- The fine-tuning datasets for the **three local personalized models** are stored in **Folders corgi, goldenretriever**, and **samoyed** respectively.

## Model Storage

The well fine-tuned models have been uploaded to huggingface, refer to the [Hugging Face wty2011 webpage]([https://huggingface.co/docs](https://huggingface.co/wty2011))

## Modifications in Hybrid Inference

We have modified the source file `pipeline_stable_diffusion.py` located in `diffusers/src/diffusers/pipelines/stable_diffusion` to support hybrid inference by adding the following parameters:
- `offloading_flag: bool = False` — Enables or disables model offloading.
- `local_flag: bool = False` — Toggles the use of a local inference model.
- `num_offloaded_step: Optional[int] = None` — Defines the number of steps to offload, if any.
- `intermediate_path: Optional[str] = None` — Specifies an optional path for saving intermediate outputs.

## Hybrid Inference Steps and Results

Details on the different offloading steps for hybrid inference and the results generated are available in the folder **fitting**.

## Algorithms

The project uses the **PER-DQN algorithm** along with comparison algorithms, details of which are located in **PER_DQN**.
