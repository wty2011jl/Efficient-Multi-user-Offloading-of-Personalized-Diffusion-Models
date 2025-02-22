from diffusers import StableDiffusionPipeline
import torch
import os

prompt1 = "A photo of sks dog in the box"  #0-20
prompt2 = "A photo of sks dog running on the grass"
prompt3 = "A photo of sks dog on the sofa"
prompt4 = "A photo of sks dog next to the car"
prompt5 = "A photo of sks dog siting on the floor"
prompt6 = "A photo of sks dog lying on a chair"
prompt7 = "A photo of two sks dogs chasing each other"
prompt8 = "A photo of sks dog sleeping on the bed"
prompt9 = "A photo of sks dog drinking water"
prompt10 = "A photo of sks dog crossing the road"

prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]


model_id_e = "path-to-save-model-collection"
# model_id_1 = "path-to-save-model-corgi"
# model_id_2 = "path-to-save-model-goldenretriever"
# model_id_3 = "path-to-save-model-samoyed"

pipe_e = StableDiffusionPipeline.from_pretrained(model_id_e, torch_dtype=torch.float16).to("cuda")
# pipe_1 = StableDiffusionPipeline.from_pretrained(model_id_1, torch_dtype=torch.float16).to("cuda")
# pipe_2 = StableDiffusionPipeline.from_pretrained(model_id_2, torch_dtype=torch.float16).to("cuda")
# pipe_3 = StableDiffusionPipeline.from_pretrained(model_id_3, torch_dtype=torch.float16).to("cuda")

for j in range(1, 11):
    sub_path = "prompt%d" %j
    # offloaded 20 step
    if j == 1:
        prompt = prompt1
    elif j == 2:
        prompt = prompt2
    elif j == 3:
        prompt = prompt3
    elif j == 4:
        prompt = prompt4
    elif j == 5:
        prompt = prompt5
    elif j == 6:
        prompt = prompt6
    elif j == 7:
        prompt = prompt7
    elif j == 8:
        prompt = prompt8
    elif j == 9:
        prompt = prompt9
    else:
        prompt =prompt10




    # for hh in range(1,10):
    #
    #     step = 20 * hh
    #
    #     for i in range(20):
    #         image = pipe_e(prompt, num_inference_steps=200, offloading_flag=True, num_offloaded_step=step, guidance_scale=7.5,
    #                      intermediate_path=r"intermediate_latents2.pth").images[0]
    #         path  = os.path.join("intermediate_result\%d" % step, sub_path)
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         image.save(os.path.join(path, "%d.png" %i))
    #
    #
    #         image = pipe_1(prompt, num_inference_steps=200, offloading_flag=True, local_flag=True, num_offloaded_step=step,
    #                      guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
    #         path = os.path.join("final_result1/%d" % step, sub_path)
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         image.save(os.path.join(path, "%d.png" %i))
    #
    #
    #         image = pipe_2(prompt, num_inference_steps=200, offloading_flag=True, local_flag=True, num_offloaded_step=step,
    #                       guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
    #         path = os.path.join("final_result2/%d" % step, sub_path)
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         image.save(os.path.join(path, "%d.png" %i))
    #
    #
    #         image = pipe_3(prompt, num_inference_steps=200, offloading_flag=True, local_flag=True, num_offloaded_step=step,
    #                       guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
    #         path = os.path.join("final_result3/%d" % step, sub_path)
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         image.save(os.path.join(path, "%d.png" %i)) #
    #

    # step = 0
    # for i in range(20):
    #
    #     image = pipe_1(prompt, num_inference_steps=200,
    #                  guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
    #     path = os.path.join("final_result1/%d" % step, sub_path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     image.save(os.path.join(path, "%d.png" %i))  #
    #
    #     image = pipe_2(prompt, num_inference_steps=200,
    #                   guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
    #     path = os.path.join("final_result2/%d" % step, sub_path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     image.save(os.path.join(path, "%d.png" %i))  #
    #
    #     image = pipe_3(prompt, num_inference_steps=200,
    #                   guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
    #     path = os.path.join("final_result3/%d" % step, sub_path)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     image.save(os.path.join(path, "%d.png" %i))  #

    step = 200
    for i in range(20):

        image = pipe_e(prompt, num_inference_steps=200,
                       guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
        path = os.path.join("final_result1/%d" % step, sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(os.path.join(path, "%d.png" % i))  #

        # image = pipe_e(prompt, num_inference_steps=200,
        #                guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
        path = os.path.join("final_result2/%d" % step, sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(os.path.join(path, "%d.png" % i))  #

        # image = pipe_e(prompt, num_inference_steps=200,
        #                guidance_scale=7.5, intermediate_path=r"intermediate_latents2.pth").images[0]
        path = os.path.join("final_result3/%d" % step, sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(os.path.join(path, "%d.png" % i))
