## In-Depth Description of Korean 자기소개서 Evaluation Model Fine-tuning:

### Project Goal: This project delves into fine-tuning a relatively small Korean Language Model (LLM) specifically for evaluating 자기소개서, a vital component of the Korean employment process. Essentially, I aim to create a model that can assess the quality and effectiveness of self-introduction essays submitted by job candidates.

### Methodology Overview:
  1. System Setup:
     -Recognizing the computational demands of training large models, I leverage Google Colab Pro with its robust 12GB RAM and VRAM configuration. This ensures efficient resource allocation and training stability.
   - Since the Colab free version has the limitations of VRAM usage, 1,000 examples with 500 epochs of training is almost the maximum threshold that can be trained.
     
2. Data Acquisition and Preparation:
   - Data Source: I get the Data from JobKorea[https://www.jobkorea.co.kr/], a widely used Korean job search platform, to harvest real-world data. This platform provides valuable insights into successful self-introduction examples and expert feedback, both crucial for model training.
   - Data Filtering: To ensure focus and relevance, I employ Selenium scraping automation to carefully filter data based on specific company size categories (large, mid-sized, foreign, and top 1000 Korean). This targeted approach helps us curate a dataset that closely mirrors the actual application scenarios.
   - Data Volume: Through this meticulous process, I accumulated a substantial dataset of 7,732 paired self-introduction and advice samples, meticulously extracted from 2,158 pages. This sizable collection provides a strong foundation for model training and evaluation.
     
3. Model Selection and Rationale:
   - EleutherAI/polyglot-ko-1.3b[https://huggingface.co/EleutherAI/polyglot-ko-1.3b]: Mu choice falls on this pre-trained Korean LLM with 1.3B parameters. Several factors underpin this decision:
      - Resource Suitability: Compared to recent colossal models with 7B-175B parameters, this model strikes a balance between parameter size and performance. This alignment is crucial for efficient fine-tuning within the constraints of personal or small company systems.
      - Fine-tuning Potential: This model's architecture demonstrates exceptional suitability for fine-tuning tasks, making it highly adaptable to the specific demands of evaluating 자기소개서.
- Balancing Performance and Resources: While larger models often boast superior performance, they also necessitate excessive computational resources. My choice prioritizes a balance between the two, enabling effective training with available resources while striving for optimal performance.
  
4. Fine-tuning Approach:
   - Parameter-Efficient Fine-Tuning (PEFT)[[https://arxiv.org/abs/2312.12148]]: Recognizing the cost implications of full fine-tuning, I adopt PEFT methodologies. These innovative techniques strategically focus on optimizing a subset of model parameters, significantly reducing computational demands and training time.
   - QLoRA Algorithm[https://arxiv.org/abs/2305.14314]: Among various PEFT options, I delve into the QLoRA algorithm. This method leverages 4-bit precision for the base model, essentially compressing information while maintaining crucial details. The gradients are then backpropagated through these compressed method into specialized "Low-Rank Adapters" (LoRAs), efficiently capturing task-specific knowledge without inflating parameter count.

5. Evaluation Strategy:
   - Human Comparison: While acknowledging the challenges of crafting an objective evaluation metric for this subjective task, I compare the fine-tuned model's outputs against human evaluations of the same self-introduction essays. This approach provides valuable insights into the model's ability to align with human judgment.
   - Baseline Comparison: Additionally, I benchmark the fine-tuned model against the base model's inferences. This comparison helps us quantify the performance improvement achieved through fine-tuning specifically for the 자기소개서 evaluation task.
   - Dataset Size Acknowledgment: I recognize the limitations of our current dataset size of 1,000 examples. Future work will involve expanding the dataset to further enhance the model's generalizability and performance.
  
Example:
---------------------------------------------------------------------------------------------------
INPUT PROMPT:

    ### Instruction:
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
     법인영업 직무의 자기 소개서에 대해 평가해주세요.

    ### context:
    본인 성격의 장단점 및 가치관을 성장과정(가족, 학창시절, 주변인과의 관계)에 기반하여 작성해 주세요. (최대 1,500자 입력가능)
    "신뢰를 만든 커뮤니케이션, 플래너를 통해 배운 계획의 중요성"
    
    제 성격의 장점은 다양한 사람과 자연스럽게 어울리는 커뮤니케이션 역량입니다.
    어릴 적 어머니의 문구점을 도운 경험과 학창시절의 임원 활동으로, 저는 친구들뿐만 아니라 어른들과도 자연스럽게 어울리는 커뮤니케이션 역량을 쌓았습니다. 그리고 수많은 팀 프로젝트를 진행하며, 선후배 팀원과 빠르게 유대감을 쌓고 팀의 분위기를 만들며 커뮤니케이션 역량을 발전시켰습니다. 그 결과, 저는 팀원에게 먼저 인정받는 리더의 신뢰뿐만 아니라 팀원들과의 인간적인 신뢰도 쌓을 수 있었습니다.
    
    한편, 저는 많은 일이 집중되었을 때 집중력이 약해지곤 합니다. 일례로, 저는 대학 시절, 전공 공부와 학회 활동, 아르바이트를 동시에 진행하며 어려움을 겪었습니다. 그때 플래너를 사용하여 일주일 단위로 계획을 세웠고, 업무를 완수할 수 있었습니다. 이후 저는 계획을 세우는 습관을 키웠고, 효율적인 시간 관리로 다양한 업무 성과를 만들 수 있었습니다.좋은점 1
    
    "스스로 기회를 만드는 사람, 130개 거절 속에서 타결해 낸 공연 후원 가능성"
    
    저는 원하는 목표를 이루기 위해 스스로 기회를 만들어 왔습니다. 이를 통해 위기 속에서도 새로운 기회를 찾는 끈기의 자세를 갖출 수 있었고, 불가능의 예상을 넘어 성과를 만들 수 있었습니다. 실제로, 저는 코피노 문제를 사회에 알리는 공연 기획 프로젝트 ‘OOOOOOO’에 참여했을 당시, 이러한 자세로 공연의 후원 가능성을 타결해 낸 경험이 있습니다.
    
    공연을 진행하기 위한 자금을 확보하는 활동을 진행했을 때의 일입니다.
    
    당시 대학생의 활동을 지원하는 기업의 CSR 사례가 많았기 때문에, 저희는 기업의 후원을 유치하는 것을 목표로 설정하였습니다. 따라서 규모와 후원 활동을 기준으로 국내 100개 기업을 선정하였고, 담당자의 연락처를 리스트 화하여 메일과 통화를 통해 공연의 후원을 요청하였습니다. 하지만 기업 대부분으로부터 예산상의 불가 또는 관련성이 없다는 거절만 돌아왔습니다.
    
    약 2주의 시간이 남았을 때, 계속된 거절로 팀 전체가 불가능한 도전이라는 것을 예감하고 좌절했습니다. 저 또한 불가능을 생각했지만, 포기하지 않고 거절의 이유를 되짚어 기업 선정에 문제가 있었다는 것을 파악했습니다. 그래서 저는 코피노 문제와 관련이 있는 사회적 기업에 다시 후원을 요청해보자고 팀원들을 설득했습니다.
    
    그 결과, 팀원들과 30개 기업을 선정하여 다시 후원 요청을 진행했습니다. 결국, 저는 한 필리핀 후원 기업으로부터 미팅 제의를 받았고, 직접 담당자로 미팅에 참여하여 약 100만 원의 후원 가능성을 타결해낼 수 있었습니다.아쉬운점 1

### advice:
    
---------------------------------------------------------------------------------------------------
### BASELINE HUMAN ADVICE:
좋은점 1 내용에 있어 특별함은 없지만 항목의 요구 사항에 맞게 작성하려고 노력한 점이 좋습니다.
아쉬운점 1 지원자는 법인 영업이라는 직무 특성을 고려하여 이 사례를 제시한 것 같습니다. 하지만 특별히 호감 가는 내용이 아니라는 점이 아쉽습니다. 지원자의 자기소개서는 금융에 대한 이야기가 별로 없고, 경험한 사례들도 제각각입니다. 금융 영업을 왜 하고 싶은 지 궁금합니다. 그리고 이 사례의 성과는 이미 필리핀 후원을 하고 있는 회사 단 한 군데 실적이 전부입니다. 특별함이 느껴지지 않습니다.

---------------------------------------------------------------------------------------------------
### BASE MODEL GENERATION - ZERO SHOT:

context:형용사 형(-y) "Yes(아니다)" 라는 대답은 NO(아니라)로 대답하셨네요?(예, 아니요 / 아닙니다) 형저는 어렸을 때부터 영어를 좋아하여 영어로 책을 읽고 영어로 일기를 쓰는 것을 생활화하였습니다. 또한, 대학 재학 기간 동안 많은 활동을 통하여 영어가 제 미래의 경쟁력이라는 생각을 하였습니다.형저는 해외 경험이 없었습니다. 하지만 제가 생각한 바를 전달할 수 있었습니다. 이를 통해 저는 학생들의 도전과 실패에서 얻는 소중한 경험을 알게 되었습니다. 좋은 경험이라고 생각합니다. 형저는 영어가 잘 표현되는 글이나 문서를 읽고, 생각을 정리하는 습관이 제게 있었습니다. 이와 같이 제가 배운 것들을 잘 전달할 수 있었습니다. 형저는 국내 대학 활동과 외국어인 영어, 그리고 사회 경험들을 다양한 방법으로 실천할 것입니다. 형저는 대학생으로서 할 수 있는 것이 많다고 생각합니다. 형저는 스스로 도전하기 위해, 국내 대학의 다양한 활동에 동참한다면 자신의 발전에 도움이 될 것이라고 생각합니다.형저는 영어를 활용한 대외활동에 많은 노력을 기울이겠습니다. 형저는 자신의 생각과 의견을 영어와 한국어로도 전달할 수 있는 준비가 갖춰져 있습니다. 형저는 외국어로서 영어와 한국어 모두 잘할 수 있는 능력이 갖추어져 있습니다. 형저는 자신의 지식과 정보를 다른 사람과 교류할 수 있는 기회가 있다고 생각합니다. 형저는 영어 뿐 만 아니라 한국어, 그리고 외국어로도 다른 사람과의 소통을 위해 최선을 다할 것입니다.형저

---------------------------------------------------------------------------------------------------
### FINTUNED MODEL GENERATION - ZERO SHOT:
   아쉬운점 1 질문의 요구사항을 반영하지 않은 무작정 추천서라는 느낌이 듭니다. 이 내용으로 인해 지원자가 왜 130개 거절을 극복할 수 있었는지에 대한 내용이 전혀 없다는 점이 아쉽습니다. 다만, 문제해결과정에서 지원자가 어떻게 리더십을 발휘하였는지를 제시한 점이 좋습니다.
    
---------------------------------------------------------------------------------------------------
### FINTUNED MODEL GENERATION - ONE SHOT:
  아쉬운점 1 문제 해결을 위해 저는 어떠한 노력을 하였고 그러한 노력을 통해서 어떠한 성과를 얻었는지에 대한 설명이 제시되어야 합니다. 또한 제시된 목표가 성취될 확률이 상대적으로 어느 정도이며 그것이 얼마나 가능한지에 대해서도 함께 검토할 필요가 있습니다. 그리고 목표했던 일을 성공적으로 완수하기 위해, 어떠한 준비 과정을 거쳐 이루어냈는지를 상세하게 설명해야 합니다.
