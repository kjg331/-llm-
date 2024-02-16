# Project Goal: This project delves into fine-tuning a relatively small Korean Language Model (LLM) specifically for evaluating 자기소개서, a vital component of the Korean employment process. Essentially, we aim to create a model that can assess the quality and effectiveness of self-introduction essays submitted by job candidates.

# Methodology Overview:
  1. System Setup:
     -Recognizing the computational demands of training large models, we leverage Google Colab Pro with its robust 12GB RAM and VRAM configuration. This ensures efficient resource allocation and training stability.
   - Since Colab free version has the limitaitons of VRAM usages, 1,000 examples with 500 epochs training is almost the maximum threshold that can be trained.
     
2. Data Acquisition and Preparation:
   - Data Source: I get the Data from JobKorea[https://www.jobkorea.co.kr/], which is the most popular job finding website in Korea. This platform provides valuable insights into successful self-introduction examples and expert feedback, both crucial for model training.
   - Data Filtering: To ensure focus and relevance, we employ Selenium scraping automation to carefully filter data based on specific company size categories (large, mid-sized, foreign, and top 1000 Korean). This targeted approach helps us curate a dataset that closely mirrors the actual application scenarios.
   - Data Volume: Through this meticulous process, we accumulate a substantial dataset of 7,732 paired self-introduction and advice samples, meticulously extracted from 2,158 pages. This sizable collection provides a strong foundation for model training and evaluation.
     
3. Model Selection and Rationale:
   - EleutherAI/polyglot-ko-1.3b[https://huggingface.co/EleutherAI/polyglot-ko-1.3b]: Mu choice falls on this pre-trained Korean LLM with 1.3B parameters. Several factors underpin this decision:
      - Resource Suitability: Compared to recent colossal models with 7B-175B parameters, this model strikes a balance between parameter size and performance. This alignment is crucial for efficient fine-tuning within the constraints of personal or small company systems.
      - Fine-tuning Potential: This model's architecture demonstrates exceptional suitability for fine-tuning tasks, making it highly adaptable to the specific demands of evaluating 자기소개서.
- Balancing Performance and Resources: While larger models often boast superior performance, they also necessitate exorbitant computational resources. My choice prioritizes a balance between the two, enabling effective training with available resources while striving for optimal performance.
  
4. Fine-tuning Approach:
   - Parameter-Efficient Fine-Tuning (PEFT)[[https://arxiv.org/abs/2312.12148]]: Recognizing the cost implications of full fine-tuning, we adopt PEFT methodologies. These innovative techniques strategically focus on optimizing a subset of model parameters, significantly reducing computational demands and training time.
   - QLoRA Algorithm[https://arxiv.org/abs/2305.14314]: Among various PEFT options, I delve into the QLoRA algorithm. This method leverages 4-bit precision for the base model, essentially compressing information while maintaining crucial details. The gradients are then backpropagated through these compressed weights into specialized "Low Rank Adapters" (LoRAs), which efficiently capture task-specific knowledge without inflating parameter count.

5. Evaluation Strategy:
   - Human Comparison: While acknowledging the challenges of crafting an objective evaluation metric for this subjective task, we compare the fine-tuned model's outputs against human evaluations of the same self-introduction essays. This approach provides valuable insights into the model's ability to align with human judgment.
   - Baseline Comparison: Additionally, we benchmark the fine-tuned model against the base model's inferences. This comparison helps us quantify the performance improvement achieved through fine-tuning specifically for the 자기소개서 evaluation task.
   - Dataset Size Acknowledgment: We recognize the limitations of our current dataset size of 1,000 examples. Future work will involve expanding the dataset to further enhance the model's generalizability and performance.
