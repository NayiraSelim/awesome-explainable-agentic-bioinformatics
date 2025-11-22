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

---

# 7. Transparent & Trustworthy AI Systems (General)


---

# 8. Transparency & Trustworthiness in Biomedical Analysis


---

# 9. Biomedical Data Accuracy & Reliability
- *"A Survey on the Role of Artificial Intelligence in Biobanking Studies: A Systematic Review"*, 2022, Diagnostics,  
  `biobanking, machine-learning, deep-learning, biomedical-data, pipelines`, [paper]  
  — Systematic review of 18 AI-based studies using global biobank datasets (UK, Qatar, Japan, Singapore), covering ML/DL tools, QC pipelines, disease prediction models, and large-scale biomedical data profiling.


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
