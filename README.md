# Improving Fake News Detection of Influential Domain via Domain- and Instance-Level Transfer
This is an official repository for "Improving Fake News Detection of Influential Domain via Domain- and Instance-Level Transfer", which has been published in COLING 2022. [Paper](https://aclanthology.org/2022.coling-1.250.pdf)
## Dataset
We construct our experimental datasets from [FakeNewsNet](https://arxiv.org/pdf/1809.01286.pdf) and [MM-COVID](https://arxiv.org/abs/2011.04088) for English (3-domain), and [MDFEND](https://github.com/kennqiang/MDFEND-Weibo21) for Chinese (9-domain).
### Requirements
```
python==3.6.13
torch==1.8.0
tranformers==4.13.0
```
### Run
We organize our code for Chinese and English datasets respectively, because some experimental settings are different between them.

For the Chinese dataset, you can refer to [README](https://github.com/ICTMCG/DITFEND/blob/main/DITFEND_ch/README.md).

For the English dataset, you can refer to [README](https://github.com/ICTMCG/DITFEND/blob/main/DITFEND_en/README.md).

## Reference
```
Qiong Nan, Danding Wang, Yongchun Zhu, Qiang Sheng, Yuhui Shi, Juan Cao, and Jintao Li. 2022. Improving Fake News Detection of Influential Domain via Domain- and Instance-Level Transfer. In Proceedings of the 29th International Conference on Computational Linguistics, pages 2834–2848, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.
```
or in bibtex style:
```
@inproceedings{nan-etal-2022-improving,
    title = "Improving Fake News Detection of Influential Domain via Domain- and Instance-Level Transfer",
    author = "Nan, Qiong  and Wang, Danding  and Zhu, Yongchun  and Sheng, Qiang  and Shi, Yuhui  and Cao, Juan  and Li, Jintao",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.250",
    pages = "2834--2848",
    abstract = "Social media spreads both real news and fake news in various domains including politics, health, entertainment, etc. It is crucial to automatically detect fake news, especially for news of influential domains like politics and health because they may lead to serious social impact, e.g., panic in the COVID-19 pandemic. Some studies indicate the correlation between domains and perform multi-domain fake news detection. However, these multi-domain methods suffer from a seesaw problem that the performance of some domains is often improved by hurting the performance of other domains, which could lead to an unsatisfying performance in the specific target domains. To address this issue, we propose a Domain- and Instance-level Transfer Framework for Fake News Detection (DITFEND), which could improve the performance of specific target domains. To transfer coarse-grained domain-level knowledge, we train a general model with data of all domains from the meta-learning perspective. To transfer fine-grained instance-level knowledge and adapt the general model to a target domain, a language model is trained on the target domain to evaluate the transferability of each data instance in source domains and re-weight the instance{'}s contribution. Experiments on two real-world datasets demonstrate the effectiveness of DITFEND. According to both offline and online experiments, the DITFEND shows superior effectiveness for fake news detection.",
}
```
