# Awesome Explainable Agentic Bioinformatics  
A curated collection of papers, tools, benchmarks, and resources on **Explainable Agentic Bioinformatics**, integrating **Large Language Models (LLMs)**, **Multi-Agent Systems (MAS)**, and **Explainable AI (XAI)** for transparent and trustworthy biomedical analysis.

Inspired by the awesome-llm-security list.  
Maintained by **Nayira Seleem**.

---

## 📌 Note  
This repository is aligned with the MSc research proposal *Explainable Agentic Bioinformatics: Leveraging Multi-Agent Systems and Large Language Models for Transparent and Trustworthy Biomedical Analysis*.  
It focuses on building a structured reference for LLMs, MAS, XAI, transparency, benchmarking, and bioinformatics workflows.

---

## 📑 Table of Contents
1. [Large Language Models (General)](#1-large-language-models-general)  
2. [LLMs in Bioinformatics](#2-llms-in-bioinformatics)  

3. [Multi-Agent Systems (General)](#3-multi-agent-systems-general)  
4. [MAS in Bioinformatics / Healthcare](#4-mas-in-bioinformatics--healthcare)  

5. [Explainable & Interpretable Models (General)](#5-explainable--interpretable-models-general)  
6. [Explainable Models in Bioinformatics / Medicine](#6-explainable-models-in-bioinformatics--medicine)  

7. [Transparent & Trustworthy AI Systems (General)](#7-transparent--trustworthy-ai-systems-general)  
8. [Transparency & Trustworthiness in Biomedical Analysis](#8-transparency--trustworthiness-in-biomedical-analysis)  

9. [Biomedical Data Accuracy & Reliability](#9-biomedical-data-accuracy--reliability)  
10. [Bioinformatics Processing & Computational Pipelines](#10-bioinformatics-processing--computational-pipelines)

---

# 1. Large Language Models (General)


---

# 2. LLMs in Bioinformatics

- *"The Large Language Models on Biomedical Data Analysis: A Survey"*, 2025, JBHI,  
  `llm, genomics, proteomics, transcriptomics, radiomics, single-cell, drug-discovery`, [paper]  
  — Comprehensive survey summarizing the applications of large language models across all major biomedical data modalities. The review covers LLM fundamentals, biomedical datasets, and frameworks, and analyzes LLM applications in genomics, proteomics, transcriptomics, radiomics, single-cell analysis, medical text mining, and drug discovery. It also discusses evaluation metrics and current challenges in applying LLMs to biomedical data analysis.

- *"Foundation Models for Bioinformatics"*, 2023,  
  `foundation-models, llms, transformers, genomics, proteomics`, [paper]  
  — Perspective review of transformer-based foundation models in bioinformatics, covering both general LLMs (e.g., ChatGPT) and bio-specialized models such as DNABERT, Geneformer, ESM, ProtGPT2, and sequence/structure foundation models. Discusses domain adaptation, prompt engineering, hallucination mitigation, and future directions for large-scale biological foundation models.

- *"HONeYBEE: Enabling Scalable Multimodal AI in Oncology through Foundation Model–Driven Embeddings"*, 2024,  
  `foundation-models, multimodal-ai, llms, oncology, tcga, embeddings`, [paper]  
  — Introduces HONeYBEE, an open-source framework that generates unified patient-level embeddings from clinical text, structured data, whole-slide pathology images, radiology scans, and molecular profiles using domain-specific foundation models and multimodal fusion. Evaluated on 11,400+ TCGA patients across 33 cancer types, HONeYBEE achieves 98.5% cancer-type classification accuracy and 96.4% precision@10 in patient retrieval. Clinical-text embeddings from general-purpose LLMs (e.g., Qwen3) outperform specialized medical models, with multimodal fusion improving survival prediction for selected cancers.

- *"Revolutionizing Personalized Medicine with Generative AI: A Systematic Review"*, 2024,  
  `generative-ai, dgms, llms, foundation-models, precision-medicine, synthetic-data`, [paper]  
  — Systematic review of deep generative models (GANs, VAEs, DGMs) and foundation models for personalized medicine, covering synthetic clinical data generation, early diagnostics, bioinformatics applications, and individualized treatment effect modeling. Highlights both the potential of GAN-based synthetic data for privacy-preserving precision medicine and current limitations in LLM-based diagnostic accuracy, outlining key research gaps and future directions.

- *"Large Language Models With Applications in Bioinformatics and Biomedicine"*, 2025, IEEE JBHI,  
  `llms, foundation-models, multimodal-biomedical-ai, molecular-modeling, interpretability`, [paper]  
  — Guest editorial summarizing 10 state-of-the-art LLM-driven advances in bioinformatics and biomedicine, covering molecular property prediction, drug–herbal interaction modeling, protein/RNA function identification, multimodal fusion, and clinical AI. Highlights emerging solutions to data scarcity (transfer learning, contrastive learning, knowledge distillation), multimodal alignment (GNNs and structural embeddings), and interpretability (attention visualization, saliency analysis, symbolic regression).

- *"Progress and Opportunities of Foundation Models in Bioinformatics"*, 2024,  
  `foundation-models, llms, transformers, genomics, proteomics, multimodal-biology`, [paper]  
  — A comprehensive survey outlining the evolution, architectures, and applications of foundation models in bioinformatics. Covers sequence- and structure-based FMs (DNABERT, Geneformer, ESM, ProtT5), multimodal models, single-cell and omics transformers, as well as methodological advances such as contrastive learning, knowledge distillation, and multi-task pretraining. Discusses limitations including data noise, model interpretability, and domain bias, and provides a roadmap for future FM development in computational biology.

- *"Benchmarking DNA Large Language Models on Quadruplexes"*, 2024,  
  `dna-llms, foundation-models, dnabert2, hyenadna, mamba-dna, caduceus, gquadruplex`, [paper]  
  — Benchmarks transformer-based (DNABERT-2), long-convolution (HyenaDNA), and state-space DNA models (MAMBA-DNA, Caduceus) for whole-genome prediction of G-quadruplexes (non-B DNA flipons). DNABERT-2 and HyenaDNA achieved top F1/MCC, while HyenaDNA recovered more distal enhancer and intronic quadruplexes. Results show that complementary FM architectures capture different regulatory structures, emphasizing the importance of model selection for genomics tasks.

- *"DomainST: Domain Knowledge-Guided Spatial Transcriptomics via LLM-Derived Gene Embeddings and Foundation Model Imaging Features"*, 2024,  
  `llms, multimodal-learning, spatial-transcriptomics, foundation-models, computational-pathology`, [paper]  
  — Proposes DomainST, a multimodal framework that uses LLMs to generate domain-aware gene embeddings and medical visual–language foundation models to extract multi-scale WSI features. A mixture-of-experts fusion module integrates gene and image modalities to enhance spatial gene expression prediction. Evaluated on three public ST datasets, DomainST outperforms SOTA models with a 6.7–13.7% PCC@50 improvement. Code: https://github.com/coffeeNtv/DomainST.

- *"Application of Artificial Intelligence Large Language Models in Drug Target Discovery"*, 2024,  
  `llms, drug-target-discovery, genomics, transcriptomics, proteomics, single-cell-omics`, [paper]  
  — Systematic review of LLM-based methods for drug target discovery, highlighting literature-driven target mining and biomolecular “language” modeling across genomics, transcriptomics, proteomics, and single-cell multi-omics. Discusses Transformer-based foundation model pretraining (masked LM, autoregressive LM), fine-tuning strategies, and biological insights enabled by LLMs, including variant pathogenicity prediction, gene expression modeling, PPI inference, and multi-omics integration for target prioritization.

- *"Distinguishing Word Identity and Sequence Context in DNA Language Models"*, 2024,  
  `dna-llms, dnabert, tokenization, sequence-context, foundation-models, knowledge-representation`, [paper]  
  — Analyzes how DNABERT learns sequence identity versus long-range context using overlapping k-mer tokens. Introduces a new token-agnostic benchmark task for evaluating DNA foundation models through non-overlapping next-token prediction, enabling unbiased assessment of contextual learning. Embedding analysis reveals that overlapping-token models primarily encode k-mer identity and struggle with larger contextual dependencies, highlighting the need for improved tokenization strategies in genomic LLMs.

- *"On Advancing Healthcare Informatics With Large Language Models"* (IEEE JBHI Guest Editorial, 2025),  
  `llms, healthcare-informatics, medical-dialogue, multimodal-llms, eeg, histopathology, hybrid-transformers`, [paper]  
  — Editorial overview summarizing five state-of-the-art LLM-based advances across healthcare informatics. Contributions include: (1) TSLLM, a Two-Stage LLM framework for biomedical information integration; (2) an LLM-enhanced multi-turn medical dialogue system improving perplexity, recall, and entity recognition; (3) an LLM–GCN hybrid model for EEG emotion recognition; (4) an LLM-guided contrastive learning framework for whole-slide image segmentation; and (5) HybridTransNet, a transformer-based multimodal diagnostic model achieving strong performance in brain tumor identification. The editorial highlights challenges in privacy, bias, interpretability, and clinical workflow integration.

- *"Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions"*, 2024,  
  `foundation-models, healthcare-fm, llms, multimodal-ai, clinical-ai`, [paper]  
  — A comprehensive survey of Healthcare Foundation Models (HFMs), covering the methodologies, data modalities, architectures, and applications of foundation models across clinical text, imaging, and multi-omics. The review analyzes key challenges, including data noise, algorithmic limitations, compute constraints, fairness, robustness, and clinical integration. It also outlines emerging directions and future opportunities for developing scalable, reliable, and generalizable HFMs for next-generation intelligent healthcare.

- *"Challenges in AI-Driven Biomedical Multimodal Data Fusion and Analysis"*, 2024,  
  `multimodal-learning, llms, biomedical-fusion, interpretability, meta-learning`, [paper]  
  — A comprehensive review of multimodal biomedical data integration, covering molecular, cellular, imaging, and EHR modalities. The paper surveys deep learning–based multimodal fusion techniques, including cross-modal attention, joint embedding learning, meta-learning, and knowledge-guided integration. It highlights challenges in privacy, fusion, and model interpretation, and discusses how large language models and pretrained foundation models can enhance multimodal biomedical analysis.

- *"Biomedical Natural Language Processing in the Era of Large Language Models"*, 2025, Annual Review of Biomedical Data Science,  
  `biomedical-nlp, llms, generative-ai, ehr, precision-health, real-world-evidence`, [paper]  
  — High-level survey discussing the evolution and future of biomedical NLP in the era of large language models. Covers foundational biomedical LLMs (BioGPT, ClinicalBERT, Med-PaLM, GatorTron), applications in clinical summarization, knowledge extraction, medical coding, real-world evidence mining, and population-level patient modeling. Highlights key challenges including hallucinations, omissions, compliance, multimodal integration (imaging + genomics), and safety within learning health systems.

- *"Foundation Models in Bioinformatics"*, 2025,  
  `foundation-models, genomics, transcriptomics, proteomics, single-cell, multimodal-fm`, [paper]  
  — A comprehensive review of recent advances in foundation models across core bioinformatics domains. The paper categorizes bioinformatics FMs into language-, vision-, graph-, and multimodal-based architectures and surveys their applications in genomics, transcriptomics, proteomics, drug discovery, and single-cell analysis. It highlights how large-scale pretraining enables robust biological representations and discusses challenges related to evaluation, interpretability, and model selection for specific downstream biological tasks.


- *"Foundation Model: A New Era for Plant Single-Cell Genomics"*, 2025,  
  `single-cell-genomics, foundation-models, scplantllm, scgpt, geneformer, multimodal-omics`, [paper]  
  — Perspective article reviewing the rise of single-cell foundation models (Geneformer, scGPT, scFoundation, GeneCompass, CellFM) and introducing scPlantLLM, the first plant-specific Transformer FM trained on large-scale plant single-cell datasets. scPlantLLM achieves strong zero-shot cell-type annotation and robust batch integration across unseen species, addressing challenges unique to plant genomics such as polyploidy and tissue-specific expression. The paper highlights future directions including multimodal integration (transcriptomics, epigenomics, imaging) and cross-scale genome modeling.

- *"CPST-GAN: Conditional Probabilistic State Transition Generative Adversarial Network With Biomedical Large Foundation Models"*, 2025,  
  `foundation-models, multimodal-fusion, imaging-genetics, gan, alzheimers-disease`, [paper]  
  — Introduces CPST-GAN, a generative framework that combines biomedical large foundation models with a conditional probabilistic state transition model to characterize Alzheimer's disease progression. High-quality imaging–genetic embeddings are extracted using pretrained biomedical FMs and fused within a GAN to model dynamic brain-region state transitions under genetic regulation. Experiments on public imaging-genetics datasets demonstrate superior AD risk prediction and evolutionary pattern mining compared to deep learning baselines.

- *"Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics"*, Nature Methods, 2024/2025,  
  `genomic-foundation-models, dna-llm, masked-language-modeling, multi-species-genomics, variant-prediction`, [paper]  
  — Introduces the Nucleotide Transformer (NT), a suite of DNA foundation models ranging from 50M to 2.5B parameters trained on the human reference genome, 3,202 human genomes, and 850 species. NT models generate context-aware nucleotide embeddings and achieve strong performance across 18 genomics benchmarks under 10-fold cross-validation, outperforming multiple foundational and supervised baselines. The study evaluates attention patterns, perplexity, scaling laws, and zero-shot variant prioritization, demonstrating the power of large-scale DNA pretraining for molecular phenotype prediction.

- *"Single-cell Foundation Models: Bringing Artificial Intelligence into Cell Biology"*, 2025,  
  `single-cell-foundation-models, scfm, scgpt, geneformer, multimodal-single-cell`, [paper]  
  — A comprehensive review of single-cell foundation models (scFMs), detailing how transformer-based, self-supervised architectures learn cell- and gene-level representations from large-scale single-cell datasets. The paper analyzes applications in downstream tasks such as cell-type classification, batch correction, trajectory inference, and regulatory network modeling. It discusses challenges including data heterogeneity, non-sequential omics structure, computational cost, and interpretability of latent embeddings, and outlines future directions for robust and scalable scFMs.

- *"Beyond Digital Twins: The Role of Foundation Models in Enhancing the Interpretability of Multiomics Modalities in Precision Medicine"*, 2025,  
  `multiomics, foundation-models, digital-twins, interpretability, precision-medicine`, [paper]  
  — A comprehensive review discussing how foundation models (FMs) improve the interpretability of multiomics data within Medical Digital Twin (MDT) systems. The paper highlights the integration of genomics, transcriptomics, proteomics, metabolomics, and epigenomics into FM-driven frameworks for precision medicine. It analyzes current challenges in multiomics fusion, biological interpretability, and computational scaling, and outlines future opportunities for FM-based MDTs in personalized treatment simulation and biological decision support.

- *"Large Language Models and Their Applications in Bioinformatics"*, 2024/2025,  
  `llm, transformer, bioinformatics, nlp, genomics, proteomics`, [paper]  
  — Introductory review summarizing the architecture and capabilities of large language models (LLMs), including GPT- and BERT-based approaches, and discussing their emerging applications across bioinformatics. The paper highlights how transformer-based LLMs are applied to genomics, proteomics, gene expression analysis, pathway analysis, and drug discovery, emphasizing their potential for handling large-scale biological data and enabling new computational insights.

- *"Evaluation of Large Language Models for Discovery of Gene Set Function"*, Nature Methods, 2025,  
  `gene-set-analysis, functional-genomics, llm, gsa, pathway-analysis`, [paper]  
  — A systematic evaluation of five LLMs (GPT-4, GPT-3.5, Gemini Pro, Mixtral, Llama2-70B) for interpreting gene sets derived from omics data. The study introduces an automated Gene Set AI (GSAI) pipeline to generate functional summaries, rationales, citations, and confidence scores for gene sets. GPT-4 accurately recovers curated GO functions in 73% of cases with calibrated confidence and yields near-zero confidence on random gene sets. Across omics gene clusters, LLMs provide specific and verifiable functional hypotheses, positioning LLMs as practical assistants for functional genomics.

- *"Benchmarking Large Language Models for Genomic Knowledge with GeneTuring"*, 2025,  
  `genomics, llm-benchmark, geneTuring, genomic-knowledge, ncbi-integration`, [paper]  
  — Introduces GeneTuring, a comprehensive 16-task benchmark with 1,600 curated genomics questions used to evaluate 48,000 answers from 10 LLM configurations. The study assesses GPT-4o, GPT-3.5, Gemini Advanced, Claude 3.5, BioGPT, BioMedLM, GeneGPT, and a custom hybrid model (SeqSnap: GPT-4o + NCBI APIs). SeqSnap achieves the highest overall performance, demonstrating that integrating LLMs with domain-specific genomic tools significantly reduces hallucinations and improves accuracy. The benchmark exposes critical limitations of current LLMs in genomic reasoning and provides a key resource for improving genomic intelligence systems.

- *"The Development Landscape of Large Language Models for Biomedical Applications"*, 2025, Annual Review of Biomedical Data Science,  
  `biomedical-llms, clinical-nlp, model-development, transformer-architectures, survey`, [paper]  
  — A comprehensive PRISMA-guided review of 82 biomedical LLMs developed since 2022. The paper analyzes model architectures (dominated by decoder-only transformers such as Llama 7B), training strategies, biomedical corpora, and applications ranging from clinical NLP to chatbots and domain-specific biomedical reasoning. It highlights challenges including privacy constraints, limited transparency in model development, and restricted data sharing, and outlines future directions toward multimodal integration and specialized biomedical LLMs.

- *Are genomic language models all you need? Exploring genomic language models on protein downstream tasks (2024)*  
A foundational study evaluating genomic foundation models (gLMs) on protein downstream tasks.  
The authors benchmark multiple Nucleotide Transformer models (50M–2.5B parameters), introduce a new 3-mer tokenization FM, and compare gLMs with protein LMs (pLMs). Results show gLMs are competitive with pLMs and that a joint genomic–proteomic FM provides superior performance. The work systematically analyzes tokenization choices, scaling behavior, representation learning, and cross-domain generalization, making it a core reference in FM development.

- *LucaOne: A Generalized Biological Foundation Model (2024)*
A unified transformer-based foundation model jointly trained on DNA, RNA, and protein sequences from 169,861 species. LucaOne integrates nucleic acid and protein languages within a single semi-supervised architecture, showing emergent understanding of the central dogma and delivering competitive performance across genomics, transcriptomics, and proteomics tasks. The model demonstrates strong few-shot generalization, cross-molecule representation learning, and applicability across diverse biological downstream analyses.

- *"Open-source Large Language Models in Action: A Bioinformatics Chatbot for the PRIDE Database"*, 2024, EMBL-EBI  
  `llm-chatbot, bioinformatics-databases, vector-search, dataset-discovery, llama2`  
  — Introduces an open-source LLM-powered chatbot integrating Llama2, ChatGLM, Mixtral, and OpenHermes to support PRIDE documentation navigation and dataset retrieval. Includes vector database indexing, API services, and Elo-based benchmarking for model evaluation, providing a modular architecture that can generalize to other bioinformatics resources.

- *"The Role of Chromatin State in Intron Retention: A Case Study in Leveraging Large-Scale Deep Learning Models"*, 2024, PLOS Computational Biology  
  `genomic-foundation-models, chromatin-state, intron-retention, DNABERT2, Sei-model`  
  — Demonstrates how large-scale genomic foundation models (Sei) encode chromatin-state information and outperform DNA language models (DNABERT-2) in predicting intron retention. The authors show that Sei embeddings help uncover transcription-factor activity and chromatin marks regulating intron retention, enabling accurate and interpretable gene-regulation modeling.

- *"Bridging artificial intelligence and biological sciences: a comprehensive review of large language models in bioinformatics"*, 2024  
  `llms, bioinformatics, survey, protein-structure, genomics, drug-discovery`  
  — A comprehensive review covering the development and applications of LLMs across core bioinformatics domains, including protein/nucleic acid structure prediction, omics analysis, biomedical literature mining, and AI-driven drug design. Discusses key challenges such as interpretability, data bias, and the future potential of cross-modal and interdisciplinary LLM integration.

---

# 3. Multi-Agent Systems (General)
- *"AbstractSwarm Multi-Agent Logistics Competition: Multi-Agent Collaboration for Improving A Priori Unknown Logistics Scenarios"*, 2024, GECCO,  
  `multi-agent-systems, agent-collaboration, simulation, logistics`, [paper]  
  — Introduces the AbstractSwarm multi-agent simulation framework and the GECCO logistics competition benchmark, providing a standardized environment for evaluating agent collaboration and adaptability in unknown logistics scenarios.


---

# 4. MAS in Bioinformatics & Healthcare
- *"Accelerating Drug Discovery: How Agentic AI and Multi-Agent Collaboration Transform BioPharma R&D"*, 2025, JISEM,  
  `agentic-ai, multi-agent-systems, drug-discovery, biopharma, automation`, [paper]  
  — Conceptual analysis of agentic AI and multi-agent collaboration frameworks that automate and optimize drug discovery pipelines, including autonomous target identification, high-throughput virtual screening, toxicity and ADMET prediction, and AI-driven adaptive clinical trial design. Highlights the role of coordinated AI agents in personalizing medicine and reducing development timelines and failure rates.

- *"Reinforcing Clinical Decision Support through Multi-Agent Systems and Ethical AI Governance"*, 2024,  
  `multi-agent-systems, clinical-decision-support, ethical-ai, transparency, icu, explainability`, [paper]  
  — Proposes a modular multi-agent architecture for ICU clinical decision support, including lab-analysis agents, vitals interpreters, contextual reasoners, prediction modules, and validation agents. Built on the eICU database, the system emphasizes transparency, autonomy, fairness, and accountability, improving interpretability and trustworthiness in AI-assisted clinical decisions.


---

# 5. Explainable & Interpretable Models (General)


---

# 6. Explainable Models in Bioinformatics / Medicine
- *"A Systematic Review of Biologically-Informed Deep Learning Models for Cancer: Fundamental Trends for Encoding and Interpreting Oncology Data"*, 2023, BMC Bioinformatics,  
  `explainable-ai, cancer, multi-omics, biological-priors, graph-neural-networks, interpretability`, [paper]  
  — Reviews 42 deep learning studies in oncology that integrate biological prior knowledge (pathways, PPIs, GO hierarchies) into neural architectures to improve biological interpretability. The survey highlights emerging explainability methods (SHAP, Grad-CAM, LRP, DeepLIFT), architecture-level constraints (sparse networks, GNN/GCN), and introduces the concept of *bio-centric interpretability* for transparent multi-omics cancer analysis.

- *"Accurate and Highly Interpretable Prediction of Gene Expression from Histone Modifications (ShallowChrome)"*, 2022,  
  `epigenomics, explainable-ai, histone-modifications, interpretable-models, chromhmm`, [paper]  
  — Introduces ShallowChrome, an interpretable feature-extraction and logistic-regression framework that models gene expression from histone modification profiles across 56 REMC cell types. Achieves state-of-the-art accuracy while enabling gene-specific regulatory interpretation and providing biologically coherent insights compared to ChromHMM chromatin state patterns.

- *"Explainable Artificial Intelligence for Omics Data: A Systematic Mapping Study"*, 2023,  
  `xai, omics, explainability, feature-relevance, visual-explanations, interpretable-models`, [paper]  
  — Systematic mapping of 405 studies (2010–2023) applying XAI to genomics, transcriptomics, proteomics, and metabolomics. Highlights dominant AI methods (neural networks, tree-based, statistical models), preferred post-hoc explainability techniques (feature relevance, visual explanations), interpretable architectures, and eight major research directions for XAI in omics.

- *"Personalized Health Monitoring Using Explainable AI: Bridging Trust in Predictive Healthcare"*, 2024,  
  `explainable-ai, clinical-prediction, attention-mechanisms, shap, personalized-medicine`, [paper]  
  — Introduces PersonalCareNet, a CNN–attention (CHARMS) deep learning framework combined with SHAP for global and patient-specific interpretability. Using MIMIC-III clinical data, the model achieves 97.86% accuracy while providing transparent local and global explanations through feature importance, force plots, and diagnostic heatmaps, enabling trustworthy real-time critical-care prediction.


---

# 7. Transparent & Trustworthy AI Systems (General)


---

# 8. Transparency & Trustworthiness in Biomedical Analysis
- *"Trust, Trustworthiness, and the Future of Medical AI: Outcomes of an Interdisciplinary Expert Workshop"*, 2025, J Med Internet Res,  
  `trustworthy-ai, ethics, transparency, stakeholder-engagement, medical-ai-governance`, [paper]  
  — Presents an interdisciplinary analysis of trust and trustworthiness in medical AI, highlighting limitations of purely technical fairness and explainability frameworks. Based on expert workshops in oncology imaging and genomics, the study emphasizes human-centered, multi-stakeholder involvement across the full AI lifecycle, showing that trust is a relational process shaped by users, institutions, and social contexts rather than a technical property of the model alone.

- *"Trustworthy AI in Digital Health: A Comprehensive Review of Robustness and Explainability"*, 2024,  
  `trustworthy-ai, robustness, explainability, digital-health, evaluation-metrics, llms`, [paper]  
  — A comprehensive review of robustness and explainability methods for trustworthy AI in digital health. Covers feature-attribution XAI (SHAP, LIME, IG), gradient-based explanations, counterfactuals, robustness against distribution shifts, privacy and fairness frameworks, and trust evaluation metrics such as validity, fidelity, and diversity. Discusses trust challenges and opportunities in the era of LLMs for clinical AI.

- *"Recommendations for Trustworthy Artificial Intelligence in Medical Imaging"*, 2024,  
  `trustworthy-ai, medical-imaging, future-ai-framework, robustness, fairness, explainability`, [paper]  
  — Translates the FUTURE-AI framework (Fairness, Universality, Traceability, Usability, Robustness, Explainability) into concrete implementation guidelines for medical imaging. Drawing on experience from five large European projects, the paper provides best-practice recommendations and an AI maturity checklist to support the development, evaluation, and deployment of clinically safe, transparent, and trustworthy imaging AI systems.

- *"AI-in-the-loop: The Future of Biomedical Visual Analytics Applications in the Era of AI"* (2024),  
  `visual-analytics, ai-in-the-loop, human-centered-ai, transparency, multimodal-foundation-models`, [paper]  
  — A forward-looking viewpoint discussing how LLMs and multimodal foundation models will reshape biomedical visual analytics workflows. The paper maps emerging AI trends onto interactive visualization pipelines, emphasizing transparency, reliability, and human-centered decision-making. It introduces the “AI-in-the-loop” paradigm, arguing that agency and responsibility must remain with human experts while AI augments exploration, interpretation, and visual reasoning in biomedical contexts.

- *"Detection of Early Parkinson’s Disease by Leveraging Speech Foundation Models"*, 2025,  
  `speech-foundation-models, parkinsons-disease, early-detection, clinical-validation, neuroimaging`, [paper]  
  — Evaluates three speech foundation models (wav2vec2.0, Whisper, SeamlessM4T) for early Parkinson’s disease detection from voice recordings. Both pretrained features and fine-tuned models are assessed, with fine-tuning achieving a new SOTA AUC of 91.35% on the ICEBERG dataset. Predictions correlate strongly with clinical scores and DaTSCAN neuroimaging markers, demonstrating the feasibility of speech-based foundation models as early, non-invasive PD biomarkers.

- *"Large Language Model–Based Critical Care Big Data Deployment and Extraction: Descriptive Analysis"*, 2025,  
  `clinical-llm, icu-gpt, data-extraction, sql-generation, critical-care-big-data`, [paper]  
  — Describes ICU-GPT, a large language model fine-tuned on intensive care datasets to enable automated SQL generation, multischema data extraction, and clinical query assistance. The system integrates LangChain, Microsoft AutoGen, Docker-based automated deployment, and web analytics tools (Metabase, Superset), allowing clinicians to deploy, query, and visualize ICU databases without programming expertise. Demonstrates how LLM-based pipelines streamline critical care data access and reduce the burden of complex clinical data processing.

- *"Leveraging Large Language Models and Knowledge Graphs for Advanced Biomedical Question Answering Systems"*, 2024  
  `biomedical-qa, knowledge-graphs, llm-reasoning, primekg, hetionet`  
  — Proposes a KBQA system that uses LLMs (LLaMA2-70B, GPT-4) to translate natural-language questions into Cypher graph queries over biomedical knowledge graphs (PrimeKG, Hetionet). LLMs then refine the retrieved answers to produce human-readable responses. Evaluated using BioASQ, the study highlights how KG structure and LLM quality jointly affect reasoning accuracy and reliability.

- *"Generating pregnant patient biological profiles by deconvoluting clinical records with electronic health record foundation models"*, 2025  
  `ehr-foundation-models, clinical-ai, proteomics-generation, fm-representations`  
  — Uses state-of-the-art EHR foundation models to generate 206 proteomic expression levels directly from patient clinical records, bypassing the need for traditional omics assays. The approach captures developmental-pathway proteins but struggles with metabolic markers, revealing biological structure in FM-derived embeddings. Demonstrates an FM-based proteomic signature for gestational diabetes, showcasing how clinical FMs can reconstruct biological states with high efficiency.

---

# 9. Biomedical Data Accuracy & Reliability
- *"A Survey on the Role of Artificial Intelligence in Biobanking Studies: A Systematic Review"*, 2022, Diagnostics,  
  `biobanking, machine-learning, deep-learning, biomedical-data, pipelines`, [paper]  
  — Systematic review of 18 AI-based studies using global biobank datasets (UK, Qatar, Japan, Singapore), covering ML/DL tools, QC pipelines, disease prediction models, and large-scale biomedical data profiling.

- *"Non-Imaging Medical Data Synthesis for Trustworthy AI: A Comprehensive Survey"*, 2022, ACM Computing Surveys,  
  `synthetic-data, trustworthy-ai, ehr, time-series, privacy, robustness`, [paper]  
  — Comprehensive survey of statistical and deep learning–based algorithms for generating synthetic non-imaging medical data (EHR, lab tests, and biosignals). Reviews evaluation metrics for utility, fidelity, and privacy, discusses open-source datasets and toolkits (GANs, VAEs, CTGAN, Synthea), and outlines key challenges for building reliable and privacy-preserving medical AI systems.


---

# 10. Bioinformatics Processing & Computational Pipelines
- *"A Multi-label Classification Model for Full-Slice Brain CT Images (SDLM)"*, 2019, ISBRA,  
  `deep-learning, cnn, gru, ct-imaging, multi-label`, [paper]  
  — Introduces SDLM, combining VGG16-based slice features with GRU sequence modeling to capture inter-slice dependencies for diagnosing nine intracranial abnormalities using full CT volumes.

- *"A Bioinformatics Assessment Indicating Better Outcomes With Breast Cancer Resident, Immunoglobulin CDR3-MMP2 Binding"*, 2023, Cancer Genomics & Proteomics, `bioinformatics, ig-repertoire, protease-binding, tcga-brca`, [paper]  
  — Utilises TCGA-BRCA WXS IG-CDR3 reads and web-tools (SitePrediction, AdaptiveMatch) to compute IG CDR3-MMP2 binding affinities and correlate higher affinity with improved overall survival.

- *"A Consensus Multi-View Multi-Objective Gene Selection Approach for Improved Sample Classification (CMVMC)"*, 2020, APBC,  
  `gene-selection, multi-omics, clustering, optimization, feature-selection`, [paper]  
  — Proposes CMVMC, a consensus multi-view multi-objective clustering-based feature selection method integrating gene expression, Gene Ontology (GO), and protein–protein interaction networks (PPIN) to identify non-redundant, biologically relevant genes for effective sample classification in human and yeast datasets.

- *"A Machine Learning Framework Integrating Multi-Omics Data to Predict Cancer-Related lncRNAs (LGDLDA)"*,  
  2020, APBC,  
  `multi-omics, lncRNA, disease-prediction, neural-networks, similarity-networks`, [paper]  
  — Proposes LGDLDA, a multi-view machine learning framework that integrates lncRNA–miRNA, lncRNA–protein, gene–disease, and disease ontology similarity matrices using nonlinear neural-network–based neighborhood aggregation to predict cancer-associated lncRNAs across gastric, colorectal, and breast cancer datasets.

- *"A Multi-Task CNN Learning Model for Taxonomic Assignment of Human Viruses"*, 2020, InCoB,  
  `deep-learning, mt-cnn, viral-taxonomy, genomic-reads, bayesian-ranking`, [paper]  
  — Proposes a multi-task CNN model combined with a naïve Bayesian ranking framework to assign human viral taxa and genomic regions from sequencing reads, outperforming Kraken2, Centrifuge, and Bowtie2 on divergent HIV-1 and SARS-CoV-2 datasets.

- *"A Systematic Bioinformatics Approach for Large-Scale Identification and Characterization of Host–Pathogen Shared Sequences"*, 2020, InCoB,  
  `host-pathogen, sequence-mining, nonamers, viral-genomics, comparative-bioinformatics`, [paper]  
  — Describes a large-scale computational pipeline to identify and characterize host–pathogen shared nonamer sequences, mapping 2430 shared peptides to 16,946 viral and 7506 human protein sequences, with detailed structural–functional insights into Flaviviridae–human interactions.

- *"A Systematic Review and Functional Bioinformatics Analysis of Genes Associated with Crohn’s Disease"*, 2022, BMC Genomics,  
  `systematic-review, gene-curation, functional-annotation, gwas, differential-expression`, [paper]  
  — Integrates 2496 PubMed abstracts, 133 GWAS Catalog genes, functional annotations (DAVID, GO, KEGG), drug–gene interactions, and expression data (GEO GSE111889) to curate and categorize 256 Crohn’s disease–associated genes, providing a comprehensive multi-source bioinformatics pipeline for disease gene prioritization.

- *"A Systematic Review of Biologically-Informed Deep Learning Models for Cancer: Fundamental Trends for Encoding and Interpreting Oncology Data"*, 2022,  
  `explainable-ai, multi-omics, graph-neural-networks, biological-priors, interpretability`, [paper]  
  — Reviews 42 deep learning studies in cancer with emphasis on multi-omics integration, biological prior knowledge encoding (pathways, PPI networks), and explainability methods such as SHAP, LIME, and Integrated Gradients, introducing the concept of bio-centric interpretability for oncology-focused DL models.

- *"A Systematic Study of Critical miRNAs on Cell Proliferation and Apoptosis Using the Shortest Path Approach"*, 2021,  
  `mirna-regulation, gene-networks, shortest-path, cancer-biology, functional-analysis`, [paper]  
  — Constructs a miRNA–gene regulatory network and applies a shortest-path graph-based method to compute the global impact of miRNAs on proliferation–apoptosis cell fate genes. Validated across breast and liver cancer datasets using DE-miRNA profiles, HMDD verification, functional module analysis, and survival analysis.

- *"A Systematic Study of Motif Pairs that May Facilitate Enhancer–Promoter Interactions"*, 2022, JIB,  
  `motif-analysis, enhancer-promoter, regulatory-genomics, tf-binding, co-occurrence`, [paper]  
  — Introduces EPmotifPair, a computational pipeline that identifies 423 TF-binding motif pairs significantly co-occurring in enhancers and promoters across seven human cell lines, enabling large-scale discovery of biologically meaningful enhancer–promoter interactions.

- *"Advances and Challenges in Bioinformatics and Biomedical Engineering: IWBBIO 2020"*, 2020, BMC Bioinformatics,  
  `bioinformatics-overview, editorial, iwbbio`, [paper]  
  — Editorial summary of five selected contributions presented at the IWBBIO 2020 conference, covering theoretical developments and practical applications across bioinformatics and biomedical engineering.

- *"SwellDB: Dynamic Query-Driven Table Generation with Large Language Models"*, 2025, SIGMOD Companion,  
  `llm-etl, data-integration, dynamic-table-generation, federated-data, bioinformatics-etl`, [paper]  
  — Introduces SwellDB, an LLM-driven data system that dynamically generates structured tables based on SQL queries and user-defined schemas. SwellDB integrates heterogeneous external sources—including web data, databases, and search engines—and synthesizes coherent, queriable tables on demand. Demonstrated across multiple domains, including bioinformatics, it enables automated ETL, federated data integration, and dynamic schema-based table construction for downstream analytical workflows.
