# Efficient-Multi-user-Offloading-of-Personalized-Diffusion-Models

## About the Training

This project utilizes Hugging Face's Dreambooth technology for training both cluster wide models and local personalized models. For further details on Dreambooth, refer to the [Hugging Face Dreambooth webpage](https://huggingface.co/docs).

### Data Sets

- The fine-tuning dataset for the **cluster wide model** is located in **Folder A**.
- The fine-tuning datasets for the **three local personalized models** are stored in **Folders B, C**, and **D** respectively.

## Model Storage

The fine-tuned models are stored as follows:
- **Cluster wide model**: Folder `a`
- **First local personalized model**: Folder `b`
- **Second local personalized model**: Folder `c`
- **Third local personalized model**: Folder `d`

## Modifications in Hybrid Inference

We have modified the source file `pipeline_stable_diffusion.py` located in `diffusers/src/diffusers/pipelines/stable_diffusion` to support hybrid inference by adding the following parameters:
- `offloading_flag: bool = False` — Enables or disables model offloading.
- `local_flag: bool = False` — Toggles the use of a local inference model.
- `num_offloaded_step: Optional[int] = None` — Defines the number of steps to offload, if any.
- `intermediate_path: Optional[str] = None` — Specifies an optional path for saving intermediate outputs.

## Hybrid Inference Steps and Results

Details on the different offloading steps for hybrid inference and the results generated are available in the folder **PAIfitting**.

## Algorithms

The project uses the **PED-DQN algorithm** along with comparison algorithms, details of which are located in **Folder B**.
