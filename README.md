# FINE-TUNE CODEGEMMA-1.1-2B WITH TEXT2SQL DATASET

## Target
> **Fine-tune CodeGemma-1.1-2b to boost the capability of transforming User's query to a valid SQL script.**

## Result
### - Evaluate by Full SQL Script Comparison
| Model                  | Accuracy |
| ---------------------- | -------- |
| Gemma2-2b-it           | xx%      |
| Fine-tuned Gemm2-2b-it | xx%      |

### - Evaluate by GPT4o
| Model                  | Accuracy |
| ---------------------- | -------- |
| Gemma2-2b-it           | xx%      |
| Fine-tuned Gemm2-2b-it | xx%      |

## Project Overview
> [!IMPORTANT]
> TODO

## Potential Areas for Future Optimization
- [ ] TODO

## References
> The code in this project refers to some references mentioned below：
- [LLM-Finetuning GitHub Repository](https://github.com/ashishpatel26/LLM-Finetuning?tab=readme-ov-file)
- [LLM Fine-tuning Chat Template](https://github.com/mst272/LLM-Dojo/tree/main/chat_template#gemma)
- [Best Way To Fine-tune Your LLM Using a T4 GPU](https://jair-neto.medium.com/best-way-to-fine-tune-your-llm-using-a-t4-gpu-part-3-3-71c7d0514aa6)
- [Conversation on Evaluation of Text-to-SQL Task](https://github.com/explodinggradients/ragas/issues/651)
- [Text-to-SQL fine-tune video on YouTube](https://www.youtube.com/watch?v=m64TTl3Pz28)
- [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one#flash-attention-2)
- [HuggingFace Developer Guides of Quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [What Rank r and Alpha To Use in LoRA in LLM ?](https://medium.com/@fartypantsham/what-rank-r-and-alpha-to-use-in-lora-in-llm-1b4f025fd133)
- [TaskType Parameter of LoRA Config](https://discuss.huggingface.co/t/task-type-parameter-of-loraconfig/52879/6)
- [What Target Modules Should We Add When Training with LoRA](https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/)
- [Difference of DataCollator between CausalLM and Seq2Seq Model](https://gitea.exxedu.com/aibot/LLaMA-Factory/src/commit/3a666832c119606a8d5baf4694b96569bee18659/scripts/cal_ppl.py)
