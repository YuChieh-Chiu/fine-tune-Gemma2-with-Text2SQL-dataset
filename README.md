# FINE-TUNE GEMMA-2-2B-it WITH TEXT2SQL DATASET

## Target
> **Fine-tune Gemma-2-2b-it to boost the capability of transforming User's query to a valid SQL script.**

## Test Records

<details>
  <summary>Test Result Version 1</summary>
  
  ### Evaluate by Full SQL Script Comparison
  > Directly compare the string content of each `Model Answer` with the `Ground Truth Answer` to see if they are exactly the same.

  | Model                   | Accuracy        |
  | ----------------------- | --------------- |
  | Gemma2-2b-it            | 3.33%           |
  | Fine-tuned Gemma2-2b-it | **33.33%**      |
  
  ### Evaluate by SOTA LLM
  > Let the SOTA LLM act as a judge, providing `SQL Context`, `Query`, `Ground Truth Answer`, and `Model Answer`. Ask the SOTA LLM to determine whether each Model Answer can achieve the same effect as the Ground Truth Answer in answering the Query.

  | Model                   | Accuracy        |
  | ----------------------- | --------------- |
  | Gemma2-2b-it            | 23.33%          |
  | Fine-tuned Gemma2-2b-it | **60.00%**      |
</details>


## Project Overview
> [!IMPORTANT]
> **Overview:** <br>
> This is a project using **QLoRA (Quantization + LoRA)** to fine-tune the Gemma-2-2b-it model, aiming to enhance its **Text-to-SQL** capabilities. <br><br>
> **Objective:** <br>
> The main goal of this project is to evaluate whether using the **PEFT (Parameters Efficiency Fine-Tuning)** method to adjust the model can significantly improve its performance in a specific domain. <br><br>
> **Results:** <br>
> In preliminary testing, the fine-tuned model demonstrated significantly better performance in Text-to-SQL tasks compared to the base model. This was true whether the evaluation was done by exact string matching or by having a state-of-the-art (SOTA) large language model (LLM) act as a judge. _The accuracy of the fine-tuned model improved by over **30%** compared to the base model._ <br><br>
> **Additional Information:** <br>
> The following resources were used in this project: <br>
>  - Dataset: **gretelai/synthetic_text_to_sql**
>  - Base Model: **google/gemma-2-2b-it**
>  - Accelerator: **GPU T4x2**

## Potential Areas for Future Optimization
- [ ] Try categorizing SQL question types, fine-tune the model on each type, and check if it leads to better performance.
- [ ] Try increasing the amount of data for fine-tuning the model and check if it results in better performance.
- [ ] Try increasing the max_steps for fine-tuning the model and check if it improves performance.

## References
> The code in this project refers to some references mentioned below：
- [LLM-Finetuning GitHub Repository](https://github.com/ashishpatel26/LLM-Finetuning?tab=readme-ov-file)
- [LLM Fine-tuning Chat Template](https://github.com/mst272/LLM-Dojo/tree/main/chat_template#gemma)
- [Finetune Gemma-2b for Text to SQL](https://medium.com/@hayagriva99999/finetune-gemma-2b-for-text-to-sql-90041abdda70)
- [Best Way To Fine-tune Your LLM Using a T4 GPU](https://jair-neto.medium.com/best-way-to-fine-tune-your-llm-using-a-t4-gpu-part-3-3-71c7d0514aa6)
- [Conversation on Evaluation of Text-to-SQL Task](https://github.com/explodinggradients/ragas/issues/651)
- [Steps By Step Tutorial To Fine Tune LLAMA 2 With Custom Dataset Using LoRA And QLoRA Techniques](https://www.youtube.com/watch?v=Vg3dS-NLUT4)
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one#flash-attention-2)
- [HuggingFace Developer Guides of Quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [What Rank r and Alpha To Use in LoRA in LLM ?](https://medium.com/@fartypantsham/what-rank-r-and-alpha-to-use-in-lora-in-llm-1b4f025fd133)
- [TaskType Parameter of LoRA Config](https://discuss.huggingface.co/t/task-type-parameter-of-loraconfig/52879/6)
- [What Target Modules Should We Add When Training with LoRA](https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/)
- [Difference of DataCollator between CausalLM and Seq2Seq Model](https://gitea.exxedu.com/aibot/LLaMA-Factory/src/commit/3a666832c119606a8d5baf4694b96569bee18659/scripts/cal_ppl.py)
