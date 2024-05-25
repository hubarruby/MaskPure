# *MaskPure: Improving Defense Against Text Adversaries with Stochastic Purification.*
This is a repository for *MaskPure*, presented at The 29th International Conference on Natural Lanugage and Information Systems, in June 2024.

By **Harrison Gietz** (Louisiana State University) and **Jugal Kalita** (University of Colorado Colorado Springs), and supported by the National Science Foundation under Grant No. 2050919.

**Abstract:**
The improvement of language model robustness, including successful defense against adversarial attacks, remains an open problem. In computer vision settings, the stochastic noising and de-noising process provided by diffusion models has proven useful for purifying input images, thus improving model robustness against adversarial attacks. Similarly, some initial work has explored the use of random noising and de-noising to mitigate adversarial attacks in an NLP setting, but improving the quality and efficiency of these methods is necessary for them to remain competitive. We extend upon methods of input text purification that are inspired by diffusion processes, which randomly mask and refill portions of the input text before classification. Our novel method, MaskPure, exceeds or matches robustness compared to other contemporary defenses, while also requiring no adversarial classifier training and without assuming knowledge of the attack type. In addition, we show that MaskPure is provably certifiably robust. To our knowledge, MaskPure is the first stochastic-purification method with demonstrated success against both character-level and word-level attacks, indicating the generalizable and promising nature of stochastic denoising defenses. In summary: the MaskPure algorithm bridges literature on the current strongest certifiable and empirical adversarial defense methods, showing that both theoretical and practical robustness can be obtained together.

**Other Notes about the Repository:**
- Section 5 of the manuscript corresponds to the defense_acc_experiments folder.
- Sectin 6 of the manuscript corresponds to the certified_robustness_experiments folder.
- The data used for obtaining results is described in Section 4.1. In addition to the description there, note that:
    - the IMDB test dataset used by TextFooler was further filtered down from 1000 entries to 776 entries, since the BERT model we utilized could not process inputs which exceeded a certain length of tokens. Hence, we filtered out sequences that exceeded this limit, which reudced the szie of the initial dataset.
    - Even then, only 100 samples were reported on for the IMDB experimental results (with one exception, see point below), since computational constraints terminated experiments early. Hence, two "summation" scripts are included in defense_acc_experiments/DeepWordBug_imdb_tests and in defense_acc_experiments/TextFooler50_imdb_tests; these are used to randomly select 100 samples form the larger results, which were then used for reporting in Table 2 of the manuscript.
    - For the "BERT (no defense)" accuracies reported on the IMDB dataset in Table 2, the scores on the full 776 sampels are used (rather than the limited 100), since this score was not computaitonally expensive to compute.
- In time for publication, the relevant experiementation code was regathered and combined from a previous private repository to form this new repository. As a result, many of the filepaths references may be innacurate; however, the code should work as intended with appropriate refactoring.
- Details on the rationale for hyperparamter selection and details for the specific fine-tuning regimine used for the BERT models was excluded for brevity and for ease of understanding the repository. This may be added in a later version of this repository.


