### This anonymized repository provides the source code of BDL framework (bias-guided debiasing learning for recommendation).


#### Overview
Recommender systems suffer from biases that negatively impact recommendation quality by causing the collected feedback to deviate from users' true preferences. Existing debiasing learning approaches, such as inverse propensity score and doubly robust, often require accurate estimations of each bias's contribution to each feedback, which is challenging in practice. Furthermore, debiasing learning impairs the model's understanding of preferences related to biases, leading to decreased accuracy in factual test environments under the recommendation policy. In this paper, we propose bias-guided debiasing learning (BDL), which progressively discovers more accurate preferences by deliberately utilizing biases. BDL consists of two components: (1) distribution alignment, which generates preference-oriented features while excluding biases, and (2) bias-guided learning, which helps to uncover hidden preferences using signals from biases. BDL is not tailored to specific biases, nor does it rely on accurate estimations of the degree to which biases affect feedback. Comprehensive experiments demonstrate the effectiveness of BDL in both factual and counterfactual test environments, supporting its superiority in finding user preferences and providing debiasing effects for various types of biases.


#### Requirements
##### Dataset
The datasets can be downloaded from the below links:
- Yahoo!R3 dataset: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=3
- Coat dataset: https://www.cs.cornell.edu/~schnabts/mnar/
- KuaiRec dataset: https://kuairec.com/
- Simulation dataset: https://github.com/DongHande/AutoDebias/tree/main/datasets/simulation

##### Software
- torch >= 1.10.1