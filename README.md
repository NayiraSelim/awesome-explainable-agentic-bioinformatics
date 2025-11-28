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

- *"The Large Language Models on Biomedical Data Analysis: A Survey"*, 2025, JBHI  
  `llm, genomics, proteomics, transcriptomics, radiomics, single-cell, drug-discovery`  
  — This comprehensive survey reviews the rapid expansion of LLMs across biomedical data domains. It summarizes LLM foundations (transformer architectures, tokenization strategies, pretraining corpora), biomedical datasets (omics, imaging, EHR), and key frameworks for model adaptation (prompt engineering, fine-tuning, domain-specific training). The survey categorizes LLM applications in genomics (variant interpretation, regulatory-element prediction), proteomics (function annotation), transcriptomics (gene-expression modeling), radiomics (image–text translation), and drug discovery (target mining, molecular representation). It further reviews evaluation metrics (perplexity, AUROC, Recall@K, F1) and highlights major challenges such as domain shift, hallucinations, limited benchmarks, clinical safety, and the lack of multimodal grounding. The survey positions LLMs as promising yet immature tools requiring rigorous biomedical alignment.

- *"Foundation Models for Bioinformatics"*, 2023  
  `foundation-models, transformers, genomics, proteomics`  
  — A perspective review on transformer-based foundation models for biological sequence and structure analysis. The paper compares general-purpose LLMs (GPT, T5) with domain-adapted models such as DNABERT, Geneformer, ESM, ProtGPT2, highlighting differences in tokenization (k-mers, amino acids), pretraining objectives (masked modeling, next-token prediction), and data sources (UniProt, ENCODE, RefSeq). It describes transfer learning strategies, prompt-based adaptation, and techniques for reducing hallucinations in biomedical reasoning. Applications span motif detection, variant pathogenicity estimation, protein stability prediction, and multi-omics integration. The review underscores the need for biologically grounded evaluation frameworks and improved interpretability tools.

- *"Revolutionizing Personalized Medicine with Generative AI: A Systematic Review"*, 2024  
  `generative-ai, dgms, llms, precision-medicine, synthetic-data`  
  — A systematic review of generative models in precision medicine covering GANs, variational autoencoders, diffusion models, and LLMs. It analyzes their use in synthetic EHR/omics data generation, early disease detection, drug-response modeling, and individualized treatment-effect prediction. The paper highlights how synthetic data can mitigate privacy concerns and support rare-disease modeling. It evaluates performance via fidelity metrics (Fréchet distance, NMSE), downstream predictive accuracy, and clinical calibration. Limitations include instability of GAN training, mode collapse, biases in generative distributions, and uncertainties in LLM-driven diagnoses. The review proposes multimodal generative FMs for integrating genomics, imaging, and longitudinal clinical data.

- *"Large Language Models With Applications in Bioinformatics and Biomedicine"*, 2025, IEEE JBHI  
  `llms, foundation-models, multimodal-biomedical-ai, molecular-modeling`  
  — A guest editorial summarizing state-of-the-art LLM-based developments in molecular biology and medicine. It highlights advances in molecular property prediction (LLM-driven molecular embeddings), drug–herb interaction modeling, protein/RNA functional prediction using transformer encoders, multimodal fusion (text + sequence + structure), and clinical AI. Emerging solutions include contrastive learning for multimodal alignment, knowledge distillation for efficiency, and attention/saliency maps for interpretability. The editorial underscores persistent challenges in data scarcity, multimodal harmonization, and transparent biological reasoning.

- *"Progress and Opportunities of Foundation Models in Bioinformatics"*, 2024  
  `foundation-models, llms, genomics, proteomics, multimodal-biology`  
  — A comprehensive survey outlining the evolution of biological foundation models across DNA, RNA, protein, and multimodal datasets. It reviews architectures such as DNABERT, Geneformer, ESM, ProtT5, and multi-omics transformers trained on heterogeneous biological corpora. Applications include sequence labeling, protein structure prediction, variant impact modeling, and gene regulatory network inference. The survey details pretraining techniques (masked-token modeling, contrastive learning, multitask learning) and evaluates performance on standard benchmarks. Challenges include noisy biological data, lack of standardized evaluation pipelines, limited interpretability, and domain bias. Future directions include sparse attention for scalability, cross-species generalization, and explainable biological embeddings.

- *"Challenges in AI-Driven Biomedical Multimodal Data Fusion and Analysis"*, 2024  
  `multimodal-learning, llms, biomedical-fusion, interpretability, meta-learning`  
  — This survey examines the rapidly growing field of multimodal biomedical AI, focusing on fusion of molecular, cellular, imaging, and EHR-based modalities. It categorizes fusion strategies into early fusion (feature concatenation), late fusion (ensemble-based), and deep fusion (cross-attention, joint embedding learning). The paper analyzes transformer-based multimodal architectures integrating omics and imaging, and highlights how LLMs can be used for knowledge-guided integration and metadata-aware representation learning. Key challenges include dataset imbalance, missing modalities, sample heterogeneity, privacy-preserving fusion, and the difficulty of interpreting cross-modal attention maps. The review proposes meta-learning and knowledge-graph–enhanced fusion for future scalable multimodal systems.

- *"Biomedical Natural Language Processing in the Era of Large Language Models"*, 2025, Annual Review of Biomedical Data Science  
  `biomedical-nlp, llms, ehr, generative-ai, clinical-text`  
  — A high-level review covering advances in biomedical NLP driven by large language models. It analyzes domain-specific LLMs (BioGPT, ClinicalBERT, Med-PaLM, GatorTron) and frontier general LLMs, focusing on tasks such as clinical summarization, diagnosis-support reasoning, medical NER, relation extraction, temporal extraction, and population-level health analytics. The survey evaluates hallucination risks, factuality issues, omission errors, privacy constraints, and challenges in aligning LLM outputs with clinical documentation standards. It highlights multimodal integration (radiology + EHR + genomics) and discusses pathways toward clinically safe, reliable, and interpretable biomedical LLMs.

- *"Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions"*, 2024  
  `foundation-models, healthcare-fm, llms, multimodal-ai, clinical-ai`  
  — A comprehensive healthcare-focused survey analyzing clinical foundation models (HFMs) across text, imaging, and structured EHR data. It evaluates architectures such as encoder–decoder FMs, multimodal transformers, and retrieval-augmented clinical LLMs. Applications include patient-triage prediction, report generation, diagnosis support, and disease risk modeling. Challenges addressed include data quality, noisy/biased clinical notes, compute cost, fairness/robustness issues, and real-time deployment in hospitals. The study outlines future opportunities including federated clinical FMs, reinforcement learning with clinician feedback (RLHF), and multimodal diagnostic agents.

- *"Bridging Artificial Intelligence and Biological Sciences: A Comprehensive Review of Large Language Models in Bioinformatics"*, 2024  
  `llms, bioinformatics, survey, protein-structure, genomics, drug-discovery`  
  — This review discusses LLM applications across protein structure prediction, RNA/DNA modeling, variant interpretation, biomedical literature mining, and AI-driven drug design. It highlights progress from traditional statistical sequence models to transformer-based biological LLMs and examines how LLMs capture structural constraints, evolutionary information, and biochemical patterns. The review assesses domain adaptation techniques (continual pretraining, instruction tuning), explains shortcomings such as hallucination, domain bias, and lack of mechanistic interpretability, and emphasizes future directions such as hybrid symbolic–neural reasoning and cross-modal biological knowledge integration.

- *"A Survey for Large Language Models in Biomedicine"*, 2024  
  `biomedicine, llms, multimodal-llms, zero-shot-learning, fine-tuning`  
  — A large-scale survey synthesizing findings from 484 biomedical AI/LLM publications. It categorizes LLM contributions into diagnostic reasoning, generative drug design, clinical decision support, biomedical NER/RE, causal knowledge mining, and personalized medicine. The study pays special attention to zero-shot and few-shot evaluation, reporting that while frontier LLMs excel at general reasoning, they underperform in fine-grained biomedical tasks requiring grounded domain knowledge. Adaptation techniques such as LoRA, adapters, prompt tuning, and multimodal alignment are reviewed. Key challenges include privacy, interpretability, faulty assumptions in training corpora, and the lack of biomedical safety benchmarks. The paper proposes federated and privacy-preserving biomedical LLM ecosystems.

- *"Multimodal Large Language Models in Health Care: Applications, Challenges, and Future Outlook"*, 2025  
  `multimodal-llms, clinical-ai, radiology, pathology, genomics, sensor-data`  
  — A comprehensive survey covering multimodal healthcare LLMs that jointly process text, imaging, signals, and omics. It reviews architectures such as text–vision transformers, radiology report generation models, pathology–omics fusion frameworks, and ICU multimodal monitoring agents. The paper evaluates alignment techniques (cross-attention, contrastive embedding, joint latent spaces) and identifies key clinical applications such as differential diagnosis, report–image consistency checking, surgical planning, and multimodal risk prediction. Challenges include data silos, high compute demands, hallucination amplification across modalities, regulatory barriers, and the absence of standardized clinical benchmarks. Future directions highlight unified hospital-scale AI agents, safety-guided multimodal alignment, and clinically interpretable embeddings.

- *"The Development Landscape of Large Language Models for Biomedical Applications"*, 2025  
  `biomedical-llms, model-development, transformer-architectures, clinical-nlp, survey`  
  — A PRISMA-guided review analyzing 82 biomedical-specific LLMs released since 2022. It maps evolution trends in model architecture (dominance of decoder-only Llama-like models), biomedical dataset construction (PubMed, PMC, EHR), tokenizer designs (subword vs. biomedical-aware tokenization), and parameter-efficient fine-tuning strategies (LoRA, adapters, prefix-tuning). The survey compares performance across tasks such as NER, QA, summarization, concept linking, and medical reasoning. Highlighted challenges include privacy-restricted corpora, reproducibility issues from undocumented training details, model bias, and inadequacies in factuality evaluation. The review calls for transparent documentation standards and clinically aligned safety benchmarks.

- *"Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics"*, Nature Methods, 2024/2025  
  `genomic-foundation-models, dna-llm, masked-language-modeling, variant-prediction`  
  — Introduces the Nucleotide Transformer (NT), a suite of DNA foundation models ranging from 50M to 2.5B parameters. NT is pretrained on the human reference genome, 3,202 human genomes, and 850 multi-species genomes using masked nucleotide modeling. The models generate context-aware nucleotide embeddings and are evaluated across 18 genomics benchmarks under 10-fold cross-validation, outperforming DNABERT, Enformer-lite, and supervised baselines. Analyses include scaling laws, attention distribution over regulatory elements, zero-shot variant prioritization, and cross-species generalization. Results show that larger NT models infer subtle regulatory patterns and achieve strong performance in promoter, enhancer, and splice-site prediction tasks.

- *"MutBERT: Probabilistic Genome Representation Improves Genomics Foundation Models"*, 2025  
  `genomics-foundation-models, snp-representation, masked-language-modeling, population-genomics`  
  — MutBERT replaces deterministic genome representation with probabilistic allele-frequency–based encoding to leverage population-scale variation. This reduces redundancy in invariant genomic regions and focuses model capacity on polymorphic areas rich in regulatory signals. MutBERT is pretrained with masked-language modeling and evaluated on variant effect prediction, enhancer annotation, chromatin-state inference, and regulatory element classification. Compared with DNABERT-2, Nucleotide Transformer, HyenaDNA, and MambaDNA, MutBERT shows superior SNP-aware modeling, improved separation of pathogenic vs. benign mutations, and stronger cross-population generalization on biobank datasets.

- *"Benchmarking DNA Large Language Models on Quadruplexes"*, 2024  
  `dna-llms, dnabert2, hyenadna, mamba-dna, g-quadruplex`  
  — A systematic benchmark evaluating transformer-based (DNABERT-2), long-convolution (HyenaDNA), and state-space (MambaDNA, Caduceus) foundation models in predicting G-quadruplexes (non-B DNA secondary structures). The benchmark uses whole-genome windows and curated quadruplex datasets, assessing performance via F1, MCC, and AUROC. Results show DNABERT-2 and HyenaDNA achieve highest F1/MCC, while HyenaDNA captures distal enhancer-associated and intronic quadruplexes missed by transformer-only models. The study reveals complementary strengths of different long-context architectures and highlights model-selection considerations for noncanonical DNA-structure prediction.

- *"Distinguishing Word Identity and Sequence Context in DNA Language Models"*, 2024  
  `dna-llms, dnabert, tokenization, sequence-context, k-mers`  
  — This analytical study investigates how DNABERT’s overlapping k-mer tokenization affects learning. The authors propose a benchmark that separates identity modeling (memorization of k-mers) from context modeling (long-range dependency capture). Using non-overlapping next-token prediction and embedding-space analyses, they show DNABERT overemphasizes k-mer identity, leading to redundancy and weakened contextual understanding. Visualization of embedding manifolds reveals strong clustering around repeated k-mers. The study motivates alternative tokenization strategies—including sparse k-mers, BPE-style genomic tokens, and hierarchical encodings—to improve biological relevance and reduce token redundancy.

- *"Large Language Models in Genomics: A Perspective on Personalized Medicine"*, 2024  
  `llms, genomics, precision-medicine, diagnostic-support`  
  — A perspective article analyzing how LLMs reshape clinical genomics workflows, particularly variant interpretation and genotype–phenotype association extraction. LLMs assist clinicians by generating evidence-based variant summaries, extracting gene–disease relationships from literature, and integrating EHR notes with genetic test results. The paper covers risk factors such as hallucinated pathogenicity claims, ancestry-driven biases, and overconfident reasoning. It contrasts LLM-based reasoning with ACMG/AMP standardized variant-classification guidelines and recommends hybrid systems integrating knowledge graphs, curated variant databases, and clinician oversight.

- *"Deep Learning for Genomics: From Early Neural Nets to Modern Large Language Models"*, 2025  
  `genomics-review, genomic-deeplearning, dna-llms, multimodal-genomics`  
  — A historical review charting the transition from early CNN/RNN genomic models to modern transformer-based genomic foundation models (DNABERT, NT, Enformer, HyenaDNA). The article emphasizes how attention, long-context modules, and multi-species pretraining enable models to capture distal regulatory interactions, 3D chromatin structure, and enhancer–promoter communication. Benchmarks such as DeepSEA, ENCODE, Basenji, and CAGI competitions are reviewed. Limitations include sparse labels, enormous sequence lengths, limited experimental validation, and interpretability challenges. Future directions include multimodal DNA–epigenome–transcriptome FMs and mechanistic hybrid modeling.

- *"NextVir: Enabling Classification of Tumor-Causing Viruses with Genomic Foundation Models"*, 2024  
  `viral-classification, genomic-foundation-models, dnabert, hyenadna, nucleotide-transformer`  
  — NextVir leverages genomic foundation models (DNABERT-S, HyenaDNA, NT) to classify viral reads by species and oncogenicity. By training on curated viral genomes and nonviral backgrounds, the system achieves strong generalization to unseen viral strains and noisy reads. Benchmarked against CNN/RNN viral classifiers, NextVir outperforms them across accuracy, F1, and MCC metrics. The method shows robustness on short read lengths (<150 bp), making it suitable for metagenomic surveillance and cancer-associated viral discovery. The study highlights how DNA FMs detect subtle viral oncogenic patterns absent from classical models.

- *"ViraLM: Empowering Virus Discovery Through the Genome Foundation Model"*, 2025  
  `viral-detection, dnabert2, metagenomics, short-contigs, genome-foundation-models`  
  — ViraLM builds on DNABERT-2 to detect viral fragments in metagenomic assemblies. Pretrained on ~50k viral genomes plus nonviral background, the model learns discriminative sequence patterns enabling accurate viral identification even in short contigs (<1 kb). The benchmark spans IMG/VR, RefSeq, and real metagenomic datasets, where ViraLM surpasses protein-based tools (geNomad, VIBRANT) and nucleotide baselines (VirRep, DeepVirFinder), achieving up to +22% F1 improvement. The system identifies novel viral clades missed by existing pipelines and highlights the advantage of FM-driven sequence embeddings for viral ecology and pathogen surveillance.

- *"The Role of Chromatin State in Intron Retention: Leveraging Large-Scale Deep Learning Models"*, 2024, PLOS Computational Biology  
  `genomic-foundation-models, chromatin-state, intron-retention, epigenomics, sei-model`  
  — This study investigates intron retention (IR) as a regulated post-transcriptional event shaped not only by sequence motifs but also by chromatin context. The authors use Sei (a large-scale epigenomic foundation model) and DNABERT-2 to evaluate IR across tissues. Sei embeddings capture enhancer-like and splicing-regulatory chromatin states that strongly correlate with retained introns. Benchmarking shows Sei outperforms DNABERT-2 in predicting tissue-specific IR patterns. Attribution analyses reveal enrichment of chromatin accessibility, H3K36me3, and splicing-factor motifs near retained introns. The work highlights the necessity of multimodal models (sequence + epigenome) to decode RNA processing mechanisms.

- *"Are Genomic Language Models All You Need? Exploring Genomic Language Models on Protein Downstream Tasks"*, 2024  
  `genomic-language-models, cross-domain-transfer, nucleotide-transformer, protein-tasks`  
  — This cross-domain generalization study evaluates whether genomic foundation models can perform protein-level tasks. Using multiple Nucleotide Transformer variants (50M–2.5B parameters), the authors test performance on protein property prediction benchmarks and compare results to protein LLMs (ESM-2, ProtT5). Surprisingly, genomic models achieve competitive results on several tasks and even outperform protein LMs when trained with a new 3-mer tokenization scheme. A unified DNA–protein FM is introduced, exhibiting superior performance to single-domain models. The findings suggest shared representational structure across molecular modalities and motivate unified biological FMs.

- *"Large Language Models and Genomics for Summarizing the Role of MicroRNA in Regulating mRNA Expression"*, 2024  
  `mirna-mrna-interactions, llms, text-mining, literature-mining, genomics`  
  — This paper introduces the MMIC corpus, the first large-scale dataset for extracting microRNA–mRNA regulatory relationships from scientific literature. Classical ML models, PubMedBERT, and Llama-2 are benchmarked for information extraction. PubMedBERT achieves the highest F1 (0.783), while Llama-2 performs strongly in zero-shot settings with high recall. Error analysis shows difficulties in resolving ambiguous gene symbols, nested relations, and indirect regulatory statements. The study demonstrates how LLM-powered text mining supports automated curation of gene-regulatory networks.

- *"Single-Cell Foundation Models: Bringing Artificial Intelligence into Cell Biology"*, 2025  
  `single-cell-foundation-models, scgpt, geneformer, cell-type-annotation, perturbation-prediction`  
  — This survey reviews transformer-based foundation models for single-cell omics, trained on millions of scRNA-seq profiles across tissues and species. Models such as scGPT, Geneformer, scFoundation, and CellFM are compared across pretraining strategies (masked-gene modeling, graph-based attention, cell–gene bipartite transformers). Applications include automated cell-type annotation, developmental trajectory inference, batch correction, perturbation-response modeling, and regulatory network inference. Core challenges include extreme data sparsity, batch effects, cross-platform variability, and the need for biologically interpretable token embeddings. The review advocates for multimodal single-cell FMs integrating scATAC, proteomics, and spatial omics.

- *"Foundation Model: A New Era for Plant Single-Cell Genomics"*, 2025  
  `plant-single-cell, scplantllm, scgpt, geneformer, cross-species-mapping`  
  — This perspective highlights the emergence of plant-focused single-cell foundation models. The authors present scPlantLLM, trained on diverse plant scRNA-seq datasets to overcome challenges specific to plant biology, including polyploidy, rigid cell walls, and high transcriptional noise. The model performs zero-shot cell-type annotation, stress-response prediction, and integration of datasets from different tissues and species. Emphasis is placed on data scarcity, species divergence, and the need for plant-specific multimodal FMs integrating epigenomic and imaging data. The paper outlines a roadmap for cross-species generalization and plant developmental modeling.

- *"A Survey for Large Language Models in Biomedicine"*, 2024  
  `llms, biomedicine, multimodal-llms, zero-shot-evaluation, clinical-ai`  
  — A large-scale PRISMA-style survey covering 484 biomedical LLM publications. The review organizes applications into clinical NLP, diagnostic reasoning, biomedical Q&A, drug discovery, and precision medicine. Particular emphasis is placed on zero-shot and few-shot performance across 137 evaluation studies, revealing strong general linguistic reasoning but variable accuracy on specialized biomedical tasks. The authors categorize adaptation strategies (full fine-tuning, LoRA, prompting, retrieval-augmentation, multimodal alignment) and analyze limitations such as hallucinations, privacy constraints, non-transparent data sources, and lack of standardized evaluation. Future opportunities include federated training, clinically aligned benchmarks, model audits, and safety-aware deployment pipelines.

- *"Integrating Important Tile Filtering and Pathology Foundation Model for Lung Cancer Mutation Prediction"*, 2024  
  `pathology-foundation-models, lung-cancer, mutation-prediction, tile-filtering, wsi-analysis`  
  — This study proposes a two-stage pipeline for mutation prediction from whole-slide histopathology images (WSIs). First, important tile filtering selects discriminative tissue regions via k-means clustering and classifier-derived probabilities. Second, the pathology foundation model UNI extracts semantic features, which are fed into an InceptionResNet prediction head. The model achieves AUC ≈ 0.85 for TP53 mutations across TCGA and independent clinical cohorts, outperforming classical CNN and MIL baselines. Interpretability maps show phenotypic correlates of genetic mutations (e.g., keratinization patterns). The work highlights the power of foundation models for linking histopathological morphology to genomic alterations.

- *"Improvements in Viral Gene Annotation Using Large Language Models and Soft Alignments"*, 2024  
  `viral-genomics, protein-annotation, soft-alignment, embeddings, llms`  
  — The authors introduce a soft-alignment approach based on LLM-derived amino-acid embeddings to improve functional annotation of viral proteins, which often lack homology signals detectable by BLAST. The method compares embedding vectors position-wise to produce a similarity map visualized in a BLAST-like interface. Evaluations on Virus Orthologous Groups and ViralZone datasets show that the embedding-alignment method recovers remote homologs missed by blastp and pooled embedding baselines. The model significantly increases annotation coverage, offering a scalable strategy for characterizing orphan viral proteins.

- *"Artificial Intelligence–Assisted Breeding for Plant Disease Resistance"*, 2024  
  `plant-disease-resistance, ai-in-breeding, multimodal-models, phenomics, llms`  
  — This review summarizes AI methods for enhancing plant disease resistance, covering image-based disease detection, phenotypic trait extraction, genomics-guided QTL prediction, and multi-omics integration. It highlights how LLMs and multimodal foundation models can fuse genomic, transcriptomic, phenomic, and environmental data to accelerate breeding decisions. Challenges include limited annotated datasets, environmental variability, cross-species divergence, and the need for interpretable predictions. Future directions include federated learning for cross-lab collaboration, synthetic data via generative models, and LLM-based pipelines for large-scale plant breeding.

- *"A Conceptual Framework for Human–AI Collaborative Genome Annotation (HAICoGA)"*, 2024  
  `genome-annotation, llm-agents, human-ai-collaboration, explainable-ai, annotation-framework`  
  — HAICoGA proposes a hybrid genome-annotation ecosystem where automated tools and LLM-based agents collaborate with expert curators. The framework includes modules for evidence aggregation (from GENCODE, Ensembl, UniProt), gene-structure proposal, function summarization, variant significance reasoning, and prioritization of ambiguous loci. LLM agents generate candidate annotations and rationales, while experts validate outputs, ensuring traceability and interpretability. The paper identifies core needs: provenance-aware pipelines, uncertainty quantification, explainable rationales, and integration with experimental validation. HAICoGA outlines the next generation of semi-autonomous genome-annotation systems.

- *"The Application of Large Language Models to the Phenotype-Based Prioritization of Causative Genes in Rare Disease Patients"*, 2024  
  `rare-diseases, phenotype-analysis, gene-prioritization, hpo, llms`  
  — This study benchmarks GPT-3.5, GPT-4, and Falcon-180B for phenotype-driven gene prioritization using Human Phenotype Ontology (HPO) terms. Across real and synthetic cohorts, LLMs achieve competitive ranking performance when candidate-gene lists contain 5–100 genes, but remain inferior to specialized HPO-based tools. LLM predictions show biases toward well-known genes (BRCA1, PTEN) and occasional hallucinated gene–phenotype links. Free-text phenotype input improves performance relative to structured formats. The authors conclude that LLMs can provide rationales and hypothesis generation but should be used as complementary assistants rather than standalone diagnostic systems.

- *"A Visual–Omics Foundation Model to Bridge Histopathology with Spatial Transcriptomics (OmiCLIP + Loki)"*, 2024  
  `multimodal-foundation-models, spatial-transcriptomics, histopathology, clip-models, omics-integration`  
  — OmiCLIP employs dual encoders (H&E patches + gene-expression sentences) with a CLIP-style contrastive objective to learn joint visual–omics embeddings. Built on OmiCLIP, the Loki platform performs tissue alignment, spatial domain annotation, cross-modal retrieval, and prediction of spatial gene expression directly from histopathology. On multiple public datasets, Loki surpasses 22 state-of-the-art methods, offering robust generalization across tissues and technologies. Interpretability analyses link histologic regions to expression signatures such as immune infiltration and tumor-stroma interfaces. The approach demonstrates the feasibility of large-scale image–omics integration.

- *"Artificial Intelligence for Multiscale Spatial Analysis in Oncology"*, 2025  
  `multiscale-ai, oncology, radiomics, pathomics, spatial-omics, foundation-models`  
  — This review synthesizes AI approaches for integrating radiology (macro-scale), pathology (micro-scale), and spatial omics (molecular scale) to characterize tumor ecosystems. The authors discuss multiscale foundation models that align information across MRI/CT features, WSI histology, and spatial transcriptomics using cross-attention, graph fusion, and contrastive learning. Applications span tumor subtyping, treatment-response prediction, and microenvironment analysis. Key challenges include data heterogeneity, lack of standardized spatial benchmarks, biological interpretability, and the high computational cost of multiscale modeling. The paper proposes federated strategies and mechanistic AI to realize clinically actionable multiscale precision oncology.

- *"Emerging AI Approaches for Cancer Spatial Omics"*, 2024  
  `spatial-omics, cancer, ai-paradigms, mechanistic-modeling, foundation-models`  
  — This review categorizes spatial omics AI into three paradigms: (1) **data-driven foundation models** (transformers, GNNs, diffusion models), (2) **constraint-based AI** incorporating biological rules (e.g., reaction–diffusion constraints, interaction priors), and (3) **mechanistic spatial models** that explicitly simulate cell–cell interactions and nutrient diffusion. The paper evaluates spatial transcriptomics and proteomics methods for tumor-microenvironment analysis, highlighting interpretability and biological grounding as essential requirements. It calls for hybrid models combining large-scale pretrained embeddings with mechanistically consistent constraints to generate testable biological hypotheses.

- *"'Bingo': A Large Language Model– and Graph Neural Network–Based Workflow for Predicting Essential Genes from Protein Data"*, 2024  
  `protein-llms, esm2, essential-gene-prediction, gnn, explainable-ai`  
  — Bingo integrates ESM-2 protein language model embeddings with a graph neural network classifier to identify essential genes across metazoans using only protein sequences. The pipeline employs adversarial training for robustness and GNNExplainer for motif- and domain-level interpretability. Benchmarks in *C. elegans*, *D. melanogaster*, mouse, and human (HepG2) demonstrate high accuracy and strong zero-shot transfer to under-annotated species. Insights reveal conserved structural motifs and domains linked to essentiality, outperforming traditional ML and GNN approaches. The study showcases the power of protein LLMs for cross-species functional genomics.

- *"Biomedical Information Integration via Adaptive Large Language Model Construction (TSLLM)"*, 2025, IEEE JBHI  
  `biomedical-entity-alignment, llm-ensemble, genetic-programming, mogp, soga`  
  — TSLLM is a two-stage system for biomedical entity alignment across heterogeneous ontologies. Stage (1): Multi-Objective Genetic Programming (MOGP) constructs a diverse population of LLM-based matchers, each capturing different embedding and prompting strategies. Stage (2): a Single-Objective Genetic Algorithm (SOGA) ensembles the best-performing matchers using learned confidence weights. The framework is evaluated on OAEI Benchmark datasets (Conference, LargeBio, Disease, Phenotype), achieving significant improvements over classical and neural baselines. A new “expert-free quality metric” is proposed for evaluating biomedical alignment without human gold standards. TSLLM demonstrates how evolutionary optimization can create robust LLM ensembles for biomedical knowledge integration.

- *"ViraLM: Empowering Virus Discovery through the Genome Foundation Model"*, 2025, Bioinformatics  
  `viral-detection, dnabert-2, genome-foundation-models, metagenomics, short-contigs`  
  — ViraLM fine-tunes DNABERT-2 on 49,929 curated viral genomes alongside challenging nonviral sequences (bacterial, archaeal, and eukaryotic host genomes). The model learns discriminative genomic-signature embeddings optimized for identifying viral contigs—especially very short ones (<1 kb), which are typically hard to classify. Benchmarks on RefSeq, IMG/VR, and real metagenomic studies show ViraLM outperforming geNomad, VIRify, VIBRANT, and DeepVirFinder with up to +22% F1 improvement on short contigs. ViraLM also identifies novel viral sequences missed by protein-based tools, demonstrating the value of DNA-level LLMs for large-scale viral discovery.

- *"DruGNNosis-MoA: Drug Mechanisms as Etiological or Palliative with Graph Neural Networks Employing a Large Language Model"*, 2025, IEEE JBHI  
  `drug-mechanisms, scibert, gnn, drug-repurposing, mechanism-of-action`  
  — DruGNNosis-MoA integrates SciBERT embeddings of drug descriptions with a graph neural network (GNN) representing drug–gene–disease interactions. The model classifies each drug’s mechanism of action (MoA) as “etiological” (disease-causal targeting) or “palliative” (symptom control). Evaluated on 2,018 FDA-approved drugs, the hybrid LLM–GNN achieves F1 ≈ 0.94, surpassing SciBERT-only and GNN-only baselines. Interpretability via attention and node-importance maps highlights pathways and protein modules that drive MoA classification. The method supports systematic drug repurposing and precision-medicine decisions.

- *"Deep Learning for Genomics: From Early Neural Nets to Modern Large Language Models"*, 2025  
  `genomics-review, deep-learning, dna-llms, regulatory-genomics, multimodal-models`  
  — This comprehensive review traces the evolution of deep learning in genomics: from early CNN/RNN models to modern transformer-based genomic foundation models such as Enformer, Nucleotide Transformer, DNABERT, and HyenaDNA. Key application domains include variant effect prediction, chromatin accessibility modeling, enhancer annotation, and 3D genome structure inference. The review outlines challenges for genomic LLMs, including long-range dependency capture, limited labeled data, and interpretability barriers. Future directions focus on multimodal integration (genome + epigenome + transcriptome), sparse attention for scaling, and biologically grounded explainability frameworks.

- *"Developing a Predictive Platform for Salmonella Antimicrobial Resistance Based on a Large Language Model and Quantum Computing (SARPLLM)"*, 2025  
  `antimicrobial-resistance, salmonella, llm-adaptation, quantum-computing, pan-genomics`  
  — SARPLLM predicts antimicrobial resistance (AMR) in *Salmonella* using pan-genomic features. A two-step feature-selection pipeline combines chi-square filtering with conditional mutual information maximization, identifying key AMR-associated genes. A Qwen2 LLM fine-tuned with LoRA performs the final AMR classification. A quantum-inspired augmentation algorithm (QSMOTEN) reduces computational complexity by compressing nearest-neighbor distance calculations from O(n) to O(log n). The system includes an online visualization platform integrating knowledge graphs, pan-genome analytics, and real-time AMR prediction. SARPLLM highlights the synergy between classical genomics, LLM adaptation, and quantum-inspired computation.

- *"Advancing Plant Single-Cell Genomics with Foundation Models"*, 2025  
  `plant-single-cell, foundation-models, generative-models, scRNA-seq, multimodal-ai`  
  — This review surveys the rise of foundation models in plant single-cell genomics. Plant scRNA-seq suffers from challenges such as rigid cell walls, sparsity, species divergence, and limited datasets. The authors examine Transformer-based models (scGPT, Geneformer, scFoundation) and how they adapt to plant biology, supporting tasks such as cell-type annotation, gene-network modeling, and stress-response mapping. The paper also covers generative models (GANs, diffusion) for synthetic single-cell data generation to address data scarcity. Future directions include multimodal integration (scATAC + imaging), species-aware tokenization, and scalable cross-species embedding transfer for crop improvement.

- *"An AI Agent for Fully Automated Multi-Omic Analyses (AutoBA)"*, 2024  
  `llm-agents, automated-analysis, rna-seq, chip-seq, spatial-transcriptomics, code-generation`  
  — AutoBA is an autonomous LLM-based agent capable of end-to-end analysis of multi-omics data (WGS, WES, RNA-seq, scRNA-seq, ATAC-seq, ChIP-seq, ST). With only three user inputs (data path, description, final objective), the agent designs pipelines, generates code, executes it, and automatically repairs errors using an Automated Code Repair (ACR) module. AutoBA integrates mainstream bioinformatics tools (FastQC, HISAT2/STAR, BWA, Salmon/Kallisto, MACS2, DESeq2/EdgeR, Seurat, ChIPseeker). Evaluations show high stability, reproducibility, and adaptability across tasks. The framework demonstrates the potential of agentic LLMs for low-code biomedical data analysis.

- *"XMolCap: Advancing Molecular Captioning Through Multimodal Fusion and Explainable Graph Neural Networks"*, 2025  
  `molecular-captioning, multimodal-fusion, gnn, llms, drug-discovery`  
  — XMolCap combines SMILES/SELFIES strings, molecular images, and graph-based features to generate chemically accurate natural-language captions. Built on a BioT5 encoder–decoder backbone, the system integrates SwinOCSR for molecular-image OCR, SciBERT for textual chemical understanding, and GIN-MoMu for structural graph encoding. The stacked fusion mechanism jointly aligns visual, textual, and graph modalities. On L+M-24 and ChEBI-20 benchmarks, XMolCap achieves state-of-the-art captioning accuracy and provides interpretable GNN-based explanations highlighting functional groups and reactive substructures. Applications include drug design, molecular education, and compound database annotation.

- *"The Rise and Potential Opportunities of Large Language Model Agents in Bioinformatics and Biomedicine"*, 2025  
  `llm-agents, autonomous-ai, multiagent-systems, drug-discovery, clinical-ai`  
  — This review analyzes LLM agents—systems that integrate LLM reasoning with planning, tool use, memory, and multi-agent coordination. Applications span multi-omics analysis, drug discovery, literature mining, laboratory automation, and patient-management workflows. Architectural components include planners, tool APIs, vector-memory stores, world models, and agent-to-agent communication. Key challenges include hallucinations during tool invocation, privacy/security concerns, temporal drift in agent memory, and lack of benchmarks for agentic evaluation. The paper envisions “AI scientist” systems capable of hypothesis generation, experiment design, and semi-autonomous biomedical research.

- *"AuraGenome: An LLM-Powered Framework for On-the-Fly Reusable and Scalable Circular Genome Visualizations"*, 2025  
  `genome-visualization, llm-agents, circular-genomics, d3-visualization`  
  — AuraGenome is an LLM-driven multiagent system that generates, edits, and scales circular genome visualizations (e.g., ring, radial, and chord diagrams) without manual coding. Seven specialized agents handle intent parsing, layout design, data parsing, D3.js code generation, validation, and explanation. A layer-aware reuse mechanism enables users to repurpose visualization components across multiple datasets. User studies show substantial improvements in speed, correctness, and usability compared to tools like Circos. AuraGenome demonstrates how LLM agents can democratize complex bioinformatics visualization tasks.

- *"Assessing the Utility of Large Language Models for Phenotype-Driven Gene Prioritization in the Diagnosis of Rare Genetic Disease"*, 2024  
  `rare-disease-diagnostics, llms, gene-prioritization, hpo, clinical-ai`  
  — This benchmark evaluates five LLMs (GPT-4, GPT-3.5, three Llama-2 variants) on phenotype-driven gene prioritization using HPO terms. Accuracy for ranking the causal gene within the top 50 is ~17% for GPT-4—still below knowledge-graph–based tools. Prompt variations (structured HPO, free-text, case summaries) reveal LLMs perform better with narrative input. RAG and few-shot prompting do not significantly improve results. Models display citation bias toward well-known disease genes (e.g., BRCA1, PTEN). Despite low diagnostic accuracy, LLMs generate interpretable rationales and phenotype–variant hypotheses, positioning them as assistants in diagnostic pipelines rather than replacements.

- *"Utilizing Omic Data to Understand Integrative Physiology"*, 2024  
  `multi-omics, integrative-physiology, nlp-in-biology, bayesian-inference`  
  — This review discusses how multi-omics (transcriptomics, proteomics, metabolomics) contributes to integrative physiology research, but also highlights limitations in reconstructing organism-level biological functions from reductionist datasets. The authors identify three progress areas: (1) user-friendly, cross-indexed omics databases, (2) Bayesian frameworks combining multi-omics evidence with physiological priors, and (3) NLP/LLM systems that mine literature to build causal networks of physiological mechanisms. The review emphasizes LLM limitations, particularly difficulty in integrating structured omics data, dealing with causality, and avoiding hallucinated mechanistic links. Future directions include multimodal FMs and causal-inference–aware LLMs.

- *"Improvements in Viral Gene Annotation Using Large Language Models and Soft Alignments"*, 2024  
  `viral-genomics, llms, soft-alignment, remote-homology, protein-annotation`  
  — The paper introduces a soft alignment framework using LLM-based amino-acid embeddings to overcome the limitations of BLAST-like methods for remote viral protein homology. The approach aligns proteins via learned embedding similarity, generating an interpretable BLAST-like heatmap. Benchmarks on Virus Orthologous Groups and ViralZone show the method uncovers remote homologs missed by blastp and pooled-embedding baselines. Significant annotation improvements are reported for structurally divergent viral proteins. The approach offers a scalable strategy for improving viral gene annotation pipelines and identifying orphan proteins.

- *"Artificial Intelligence–Assisted Breeding for Plant Disease Resistance"*, 2024  
  `plant-disease-resistance, phenomics, llms, multimodal-models, genomic-selection`  
  — This review highlights AI-driven strategies for plant disease resistance, including phenomics-based image classifiers, multi-omics integration for quantitative trait modeling, and genomic selection using machine learning. The authors discuss how LLMs and foundation models can unify genomic, transcriptomic, phenotypic, and environmental variables, enabling prediction of resistance traits and faster breeding cycles. Limitations include sparse plant omics datasets, strong environmental effects, limited interpretability, and low cross-species transfer. Future visions include LLM-driven collaborative breeding platforms, federated training across field stations, and synthetic data augmentation via generative models.

- *"A Conceptual Framework for Human–AI Collaborative Genome Annotation (HAICoGA)"*, 2024  
  `genome-annotation, llm-agents, explainable-ai, interactive-ml`  
  — HAICoGA outlines a human–AI collaboration paradigm for scalable genome annotation. LLM agents propose gene structures, function summaries, and variant interpretations by aggregating evidence from databases such as Ensembl, GENCODE, and UniProt. Human experts validate and refine the suggestions, forming an iterative feedback loop. The framework emphasizes provenance tracking, rationales with uncertainty estimates, and interpretable decision-making. HAICoGA addresses challenges in data fragmentation, limited experimental annotations, and lack of transparency in current annotation tools, representing a pathway toward semi-autonomous, expert-supervised LLM annotation systems.

- *"A Visual–Omics Foundation Model to Bridge Histopathology with Spatial Transcriptomics (OmiCLIP + Loki)"*, 2024  
  `visual-omics, clip-models, spatial-transcriptomics, histopathology, multimodal-foundation-models`  
  — OmiCLIP introduces dual encoders—one for H&E histopathology patches and one for spatial transcriptomics (ST) gene-expression sentences—trained using a CLIP-style contrastive objective. The follow-up platform Loki leverages these embeddings for tissue alignment, spatial domain segmentation, and cross-modal retrieval. Loki can also predict gene expression directly from histology images. Evaluated across large public and in-house ST datasets, OmiCLIP/Loki outperforms 22 competing models in clustering accuracy, gene-expression prediction (PCC), and cell-type domain resolution. Interpretability analysis shows alignment between histologic morphology and transcriptomic signatures, enabling cross-modality biological insights.

- *"Artificial Intelligence for Multiscale Spatial Analysis in Oncology"*, 2025  
  `multiscale-ai, oncology, radiomics, pathomics, spatial-omics, tumor-microenvironment`  
  — This review integrates AI advances across three spatial scales in oncology: radiology (macro), histopathology (micro), and spatial omics (molecular). The paper discusses multiscale foundation models that align information across MRI/CT radiomics, WSI pathomics, and ST/IMC spatial omics through cross-attention, graph fusion, and multimodal contrastive learning. Applications include tumor subtyping, immune-microenvironment characterization, and therapy-response prediction. Core challenges include cross-platform heterogeneity, lack of unified annotation standards, computational cost, and biological interpretability. The authors propose mechanistic AI and federated training pipelines for clinically deployable multiscale oncology systems.

- *"Emerging AI Approaches for Cancer Spatial Omics"*, 2024  
  `spatial-omics, cancer, foundation-models, mechanistic-modeling, interpretable-ai`  
  — This review categorizes spatial-omics AI into three paradigms: (1) data-driven foundation models (transformers, GNNs, VAEs, diffusion models), (2) knowledge/constraint-based AI integrating biological priors such as ligand–receptor interactions and reaction–diffusion physics, and (3) mechanistic spatial models simulating cell–cell interactions and nutrient gradients. The authors summarize how each paradigm is applied to spatial transcriptomics and proteomics for tumor-microenvironment analysis. A key theme is interpretability—linking spatial gene-expression patterns to biological processes. They advocate for hybrid mechanistic–data-driven AI capable of generating testable hypotheses.

- *"'Bingo': A Large Language Model– and Graph Neural Network–Based Workflow for Predicting Essential Genes from Protein Data"*, 2024  
  `essential-genes, protein-llms, gnn, esm2, explainable-ai`  
  — Bingo integrates ESM-2 protein embeddings with a graph neural network classifier to predict essential genes using only protein sequences. The workflow includes adversarial training for robustness and GNNExplainer for structural/motif-level biological interpretability. The model achieves high cross-species accuracy in *C. elegans*, *D. melanogaster*, mouse, and human (HepG2), with strong zero-shot performance on poorly annotated species. Interpretability analyses highlight conserved protein domains and motifs associated with essentiality. Bingo outperforms classical ML and standalone GNN methods, demonstrating the potential of protein LLMs for functional genomics.

- *"Biomedical Information Integration via Adaptive Large Language Model Construction (TSLLM)"*, 2025, IEEE JBHI  
  `entity-alignment, llm-ensembles, evolutionary-search, mogp, soga`  
  — TSLLM constructs adaptive LLM ensembles for biomedical entity-alignment tasks. Multi-Objective Genetic Programming (MOGP) evolves diverse LLM-based matchers (varying prompts, embeddings, and similarity metrics), while a Single-Objective Genetic Algorithm (SOGA) determines optimal ensembling weights. The framework is evaluated on OAEI datasets—Conference, LargeBio, Disease, Phenotype—achieving superior matching precision and recall over rule-based and deep-learning baselines. TSLLM introduces an “expert-free” metric enabling automatic evaluation of entity-alignment quality without ground-truth labels. This work highlights how evolutionary strategies can optimize LLM integration in biomedical knowledge graphs.

- *"ViraLM: Empowering Virus Discovery through the Genome Foundation Model"*, 2025, Bioinformatics  
  `viral-discovery, genome-foundation-models, dnabert-2, metagenomics, short-contigs`  
  — ViraLM is a viral-detection framework built on DNABERT-2 and trained on nearly 50,000 curated viral genomes plus challenging nonviral sequences. The model learns discriminative genomic signatures enabling detection of viral contigs — especially ultrashort (<1 kb) and low-diversity sequences common in metagenomic data. Benchmarks on RefSeq, IMG/VR, and real metagenomes show up to +22% F1 improvement over protein-based tools (geNomad, VIBRANT, VirSorter2) and DNA-based baselines (VirRep, DeepVirFinder). ViraLM also discovers novel viral candidates missed by protein-homology workflows. Its robustness across hosts and sequencing platforms demonstrates the power of DNA-level LLMs for scalable viral surveillance.

- *"DruGNNosis-MoA: Drug Mechanisms as Etiological or Palliative With Graph Neural Networks Employing a Large Language Model"*, 2025, IEEE JBHI  
  `drug-mechanisms, scibert, gnn, drug-repurposing, mechanistic-interpretation`  
  — This study integrates SciBERT embeddings with a drug–gene–disease graph neural network to classify drugs as “etiological” (targeting root causes) or “palliative” (modifying symptoms). Three methodological variants are compared: SciBERT alone, GNN alone, and the integrated LLM–GNN model. The hybrid model achieves the best F1-score (~0.94) across 2,018 FDA-approved drugs. Interpretability tools (attention maps, node-importance scores) highlight mechanistic pathways that drive classification decisions. The work demonstrates the utility of foundation-model embeddings combined with biological knowledge graphs for explainable MoA characterization and precision-medicine repurposing.

- *"Deep Learning for Genomics: From Early Neural Nets to Modern Large Language Models"*, 2025  
  `genomics-review, dna-llms, enformer, nucleotide-transformer, multimodal-genomics`  
  — This review traces 30+ years of genomic deep learning, progressing from early multilayer perceptrons through CNN/RNN architectures to present-day genomic foundation models such as Enformer, Nucleotide Transformer, DNABERT, HyenaDNA, Caduceus, and MambaDNA. Key application domains include variant-effect prediction, enhancer/chromatin modeling, promoter classification, and 3D genome inference. The survey highlights core challenges: modeling long-range dependencies, multi-omics integration, interpretability, label scarcity, and sequencing biases. Future directions include multimodal genomic FMs (DNA + epigenomics + transcriptomics), sparse/linear attention for 1M+ token contexts, large cross-species pretraining, and experimentally guided interpretability for regulatory genomics.

- *"Advancing Plant Single-Cell Genomics with Foundation Models"*, 2025  
  `single-cell-genomics, foundation-models, plant-genomics, generative-models, multimodal-omics`  
  — This review analyzes how foundation models (FMs) and deep-learning architectures are transforming plant single-cell genomics. Plant scRNA-seq poses challenges including cellular dissociation difficulty, extreme sparsity, species divergence, and limited dataset availability. The authors examine transformer-based FMs such as GPT-like autoregressive models, BERT-style masked language models, and specialized architectures (scGPT, Geneformer, scFoundation), highlighting how these models improve cell-type annotation, developmental trajectory inference, and gene regulatory network modeling.  
  — The paper emphasizes multimodal integration, demonstrating how FMs combine scRNA-seq with scATAC-seq, proteomics, and spatial transcriptomics to build unified, biologically meaningful embeddings. A major focus is placed on generative approaches—GANs and diffusion models—which enable high-fidelity synthetic plant single-cell data generation, reduce dropout artifacts, and address class imbalance in rare cell populations.  
  — The review also introduces plant-focused foundation models such as scPlantLLM, designed for polyploid genomes, species-specific gene families, and cross-species transfer. scPlantLLM demonstrates strong zero-shot cell-type annotation and stress-response prediction across diverse plant species. The authors outline future directions including multimodal plant FMs, species-aware tokenization, and large-scale cross-species pretraining to accelerate discoveries in plant development, stress resilience, and crop improvement.

- *"An AI Agent for Fully Automated Multi-Omic Analyses (AutoBA)"*, 2024  
  `llm-agents, multi-omics, automated-analysis, code-generation, bioinformatics-pipelines`  
  — AutoBA is an autonomous LLM-driven agent designed to perform fully automated multi-omic bioinformatics analyses with minimal user input. The system requires only three inputs—data path, data description, and analysis objective—and then automatically generates analysis plans, writes code, executes pipelines, debugs errors, and performs downstream interpretation. Built to address challenges in bioinformatics reproducibility, pipeline variability, and the high training cost for wet-lab researchers, AutoBA integrates multiple LLM backends (online and local) to ensure data security and privacy. Its Automated Code Repair (ACR) mechanism enhances reliability by identifying and fixing execution failures during pipeline generation. AutoBA supports a wide range of omics data types including WGS, WES, RNA-seq, scRNA-seq, ATAC-seq, ChIP-seq, and spatial transcriptomics, and leverages mainstream tools such as FastQC, Trimmomatic, HISAT2/STAR/BWA, Salmon/Kallisto, MACS2, DESeq2/EdgeR, Seurat, and ChIPseeker. Benchmarks across diverse real-world multi-omic datasets show strong stability, adaptability, and performance exceeding general-purpose LLMs (e.g., ChatGPT) and online analysis services. AutoBA represents a major advancement in LLM-based autonomous bioinformatics, enabling low-code, reproducible, and scalable end-to-end analyses adaptable to emerging tools and methodologies.

- *"XMolCap: Advancing Molecular Captioning Through Multimodal Fusion and Explainable Graph Neural Networks"*, 2025  
  `molecular-captioning, multimodal-fusion, graph-neural-networks, llms, drug-discovery`  
  — XMolCap introduces a multimodal, explainable molecular captioning framework that integrates three complementary molecular representations: molecular images, SMILES/SELFIES strings, and graph-based structural information. Built on a BioT5 encoder–decoder backbone, the system leverages specialized modules including SwinOCSR for molecular-image OCR, SciBERT for chemical language understanding, and GIN-MoMu for graph-based structural encoding. A stacked multimodal fusion mechanism jointly aligns visual, textual, and graph embeddings to generate accurate and chemically grounded natural-language captions. XMolCap achieves state-of-the-art performance on two benchmark datasets (L+M-24 and ChEBI-20), outperforming several strong baselines. The framework provides interpretable, functional group–aware explanations via graph-based attention, highlighting key substructures and molecular properties that influence caption generation. The tool is publicly available for reproducibility and supports local deployment. XMolCap demonstrates the potential of multimodal LLM–GNN integration for interpretable molecular representation learning, with applications in drug discovery and chemical informatics.

- *"The Rise and Potential Opportunities of Large Language Model Agents in Bioinformatics and Biomedicine"*, 2025  
  `llm-agents, autonomous-ai, multi-agent-systems, bioinformatics, biomedicine`  
  — This review provides a comprehensive overview of LLM agents—systems that extend large language models with capabilities for reasoning, planning, tool use, and autonomous task execution. The authors outline the technical foundations of LLM agents, including core architectural components (planners, memory modules, tool APIs, environment interfaces), multi-agent collaboration modes, and enabling technologies such as retrieval augmentation, function calling, and tool orchestration. The paper examines diverse applications across bioinformatics and biomedicine, including automated multi-omics analysis, drug discovery pipelines, chemical reasoning, clinical decision support, diagnostic triage, and personalized health management. It also highlights the limitations of current LLM agents: framework scalability, tool orchestration complexity, privacy and data security concerns, model hallucinations, interpretability challenges, lack of real-time knowledge updates, and ethical/regulatory risks in clinical settings. Future directions include standardized open-source ecosystems for biomedical agents, human–AI collaborative paradigms, secure agent frameworks, continuous knowledge-refresh pipelines, and the emergence of “AI scientist” multi-agent systems capable of hypothesis generation, experiment planning, and autonomous biomedical reasoning. This review positions LLM agents as a transformative next step beyond conventional LLMs, with significant potential to accelerate precision medicine, biomedical research, and computational biology.

- *"AuraGenome: An LLM-Powered Framework for On-the-Fly Reusable and Scalable Circular Genome Visualizations"*, 2025  
  `genome-visualization, llm-agents, circular-genomics, visualization-frameworks, multiagent-systems`  
  — AuraGenome introduces an LLM-driven multiagent framework for generating reusable, scalable circular genome visualizations without manual scripting. The system addresses limitations of traditional tools such as Circos, which require complex configuration and iterative parameter tuning. AuraGenome employs seven specialized LLM-powered agents responsible for intent recognition, semantic parsing, layout design, D3.js code generation, validation, refinement, and explanation, enabling natural-language-driven visualization creation. Built atop this semantic multiagent workflow, the interactive visual analytics system supports multilayered circular layouts—including ring, radial, and chord representations—to visualize genomic features such as structural variants, chromosomal interactions, and regulatory elements. A layer-aware reuse mechanism allows users to adapt and repurpose visualization components across tasks, improving efficiency and narrative report generation. Validation across two real-world case studies and a comprehensive user study demonstrates significant improvements in usability, speed, and flexibility compared to traditional circular-genomics visualization pipelines. AuraGenome highlights the potential of LLM agents to democratize complex genomic visualization tasks by combining natural-language interfaces with automated, high-quality D3-based visualization generation.

- *"Assessing the Utility of Large Language Models for Phenotype-Driven Gene Prioritization in the Diagnosis of Rare Genetic Disease"*, 2025  
  `rare-disease-diagnosis, gene-prioritization, hpo, llms, clinical-genomics`  
  — This study systematically evaluates five LLMs—GPT-3.5, GPT-4, Llama2-7B, Llama2-13B, and Llama2-70B—for phenotype-driven gene prioritization, a core task in diagnosing rare genetic disorders. The authors assess performance across task completeness, gene ranking accuracy, and adherence to required output structures using multiple prompting strategies, phenotypic input formats (free text vs. standardized HPO concepts), and task difficulty levels. Despite improvements with larger models and more sophisticated prompts, GPT-4 achieves only 17% accuracy in identifying the causal gene within the top 50 predictions—substantially lower than traditional tools such as Phenomizer, Exomiser, AMELIE, and Phen2Gene, which rely on curated phenotype–gene knowledge graphs. The study finds that free-text input yields better-than-random predictions but remains slightly inferior to structured HPO input. Neither retrieval-augmented generation (RAG) nor few-shot prompting improves performance, and complex prompts reduce output-structure compliance. Bias analyses reveal a strong preference for highly studied genes (e.g., BRCA1, TP53, PTEN). Using a post-2023 dataset confirms the robustness of findings and reduces concerns about training-data leakage. The study concludes that while LLMs generate coherent rationales and can support hypothesis generation, they are not yet reliable replacements for dedicated phenotype-gene prioritization tools in clinical genomics.

- *"Utilizing Omic Data to Understand Integrative Physiology"*, 2024  
  `multi-omics, integrative-physiology, nlp-in-biology, bayesian-inference, llms`  
  — This review analyzes the challenges and opportunities in integrating large-scale omic data—particularly protein mass spectrometry and next-generation sequencing—with traditional hypothesis-driven physiology to understand organism-level biological mechanisms. The author summarizes key omic techniques relevant to physiological research and highlights three major advancements enabling integrative physiology: (1) development of cross-indexed, user-friendly omic databases that democratize access and unify heterogeneous datasets; (2) application of Bayesian frameworks to combine multi-omics evidence with mechanistic insights from classical physiology, enabling probabilistic reasoning about biological processes; and (3) use of natural language processing to mine literature and construct causal graphs that represent physiological pathways in a structured, machine-readable form. The review discusses emerging applications of large language models, including their potential to support literature synthesis and hypothesis generation, but also emphasizes current limitations such as hallucination, inability to integrate structured physiological datasets, and challenges in generating mechanistically accurate causal inferences. Overall, the paper provides a roadmap for combining omics, computational inference, and NLP/LLM technologies to advance whole-organism physiological understanding.

- *"Nicheformer: A Foundation Model for Single-Cell and Spatial Omics"*, 2025  
  `single-cell-foundation-models, spatial-omics, transformer-models, multimodal-pretraining, spatial-transcriptomics`  
  — Nicheformer is a transformer-based foundation model trained jointly on dissociated single-cell and spatial transcriptomics data to learn spatially informed cellular representations at scale. The authors curate SpatialCorpus-110M, a massive dataset containing over 110 million human and mouse cells—including 57M dissociated and 53M spatially resolved cells across 73 tissues—and pretrain Nicheformer using self-supervision to integrate both molecular expression and spatial context. By incorporating modality, organism, and assay tokens, the model learns unified representations that encode microenvironmental structure. Nicheformer achieves strong performance in linear probing and fine-tuning across newly designed downstream tasks, particularly in spatial composition prediction and spatial label prediction, outperforming existing foundation models such as Geneformer, scGPT, UCE, CellPLM, scVI, and PCA-based embeddings. Importantly, models trained only on dissociated data fail to capture microenvironmental complexity, underscoring the need for spatial-aware pretraining. Nicheformer enables accurate inference of spatial context for dissociated scRNA-seq datasets, effectively transferring spatial information from spatial transcriptomics data. This work establishes a next-generation foundation model paradigm for robust spatially informed representation learning in single-cell biology.

- *"Steering Veridical Large Language Model Analyses by Correcting and Enriching Generated Database Queries: First Steps Toward ChatGPT Bioinformatics"*, 2025  
  `llm-accuracy, bioinformatics-assistants, database-query-correction, rag, llm-steering`  
  — This study examines the limitations of ChatGPT as a bioinformatics assistant, revealing consistent problems in data retrieval, silent hallucinations, incorrect sequence manipulations, API misuse, and flawed code generation. To address these issues, the authors introduce **NagGPT**, a middleware system placed between LLMs and genomics databases that intercepts, corrects, and enriches LLM-generated queries. NagGPT validates and fixes malformed database requests, synthesizes large responses into concise snippets, and injects corrective comments back into the LLM prompt to steer reasoning. A companion custom GPT—**GenomicsFetcher-Analyzer (GFA)**—integrates ChatGPT with NagGPT, enabling dynamic retrieval of authoritative data from major genomics resources (NCBI, Ensembl, UniProt, WormBase, FlyBase) and execution of real bioinformatics software through generated Python code. Despite partial mitigations—including handling identifier confusion, improving API call consistency, and reducing hallucinated operations—the authors find that silent errors still occur, requiring user oversight and manual debugging. The work highlights significant challenges in using unmodified LLMs for scientific workflows but demonstrates a viable path toward veridical, tool-augmented LLM systems for future bioinformatics assistants.

- *"FHG-GAN: Fuzzy Hypergraph Generative Adversarial Network With Large Foundation Models for Alzheimer’s Disease Risk Prediction"*, 2025  
  `alzheimer-risk-prediction, multiomics, fuzzy-hypergraphs, generative-adversarial-networks, foundation-models`  
  — FHG-GAN is a fuzzy hypergraph–based deep learning framework designed to integrate multiomics data for Alzheimer’s disease (AD) risk prediction and mechanistic insight. The method begins by formulating a mathematical model of fuzzy structural entropy propagation, representing AD progression as topological evolution within fuzzy hypergraphs that encode high-order, uncertain associations among brain regions and genes. Large foundation models (BrainLM for fMRI and Nucleotide Transformer for SNPs) generate high-quality feature embeddings, mitigating noise and inconsistencies in heterogeneous biomedical data. These embeddings are used to construct brain-region–gene fuzzy hypergraphs. The proposed FHG-GAN employs fuzzy hypergraph convolutional layers within its generator to model disease evolution patterns, while the discriminator assesses real versus generated hypergraphs. Across multiple datasets, FHG-GAN outperforms advanced baselines in AD risk prediction, multiomics feature fusion, and evolutionary pattern discovery. It accurately extracts pathogenic brain lesions and risk genes, offering interpretable insights into disease mechanisms and supporting earlier diagnosis of AD-like neurodegenerative disorders.

- *"A Foundational Large Language Model for Edible Plant Genomes (AgroNT)"*, 2025  
  `plant-genomics, dna-foundation-models, crop-improvement, regulatory-prediction, zero-shot-variant-scoring`  
  — AgroNT is a transformer-based foundational DNA language model trained on reference genomes from 48 plant species, with a strong emphasis on edible and agriculturally significant crops. The model leverages large-scale self-supervised training to learn sequence representations directly from genomic DNA, enabling accurate prediction of diverse regulatory and functional genomic features without relying on extensive labeled datasets. AgroNT achieves state-of-the-art performance across tasks including regulatory element annotation, promoter/terminator strength prediction, tissue-specific gene expression inference, and functional variant prioritization. Demonstrating strong zero-shot capabilities, the model accurately predicts the impact of variants even in understudied “orphan crops.” The authors perform large-scale in silico saturation mutagenesis on cassava, evaluating over 10 million mutations to map their regulatory effects, providing a valuable public resource for variant characterization. AgroNT is released on HuggingFace, and the study introduces the Plants Genomic Benchmark (PGB), a comprehensive multi-task benchmark for evaluating deep-learning approaches in plant genomics. The results highlight AgroNT’s utility for advancing crop genomic improvement, regulatory annotation, and model-guided genome editing across diverse plant species.

- *"Optimized Biomedical Entity Relation Extraction with Data Augmentation and Classification Using GPT-4 and Gemini"*, 2024  
  `biomedical-ner, relation-extraction, llm-augmentation, gpt4, gemini, bionlp`  
  — This work proposes a hybrid large-language-model–enhanced pipeline for biomedical named entity recognition (NER) and relation extraction (RE), addressing challenges such as multistage prediction, ontology-dependent entity identifiers, unbalanced datasets, and cross-sentence relations. The approach integrates GPT-4 for synthetic data augmentation, Gemini for generating enriched relation-aware outputs, and an ensemble of fine-tuned BioNLP–PubMedBERT classifiers for final prediction. The system is designed to overcome the limitations of existing models: AIONER lacks identifier normalization, BERT-GT does not perform NER, and BioREX omits novelty prediction. Leveraging LLM-generated augmented data improves robustness to rare relation types and enhances coverage beyond sentence-level interactions. Experimental results on the **BioCreative VIII BioRED** benchmark show consistent gains in precision, recall, and F1, including improved detection of relation types (“binding,” “association,” “drug interaction,” etc.) and prediction of the “novelty” attribute. The LLM-augmented framework demonstrates that GPT-4 and Gemini can meaningfully enhance RE performance when combined with domain fine-tuning, but still require curated classification models for reliable biomedical extraction.

- *"Leveraging Large Language Models to Predict Antibiotic Resistance in Mycobacterium tuberculosis (LLMTB)"*, 2025  
  `antimicrobial-resistance, mtb-genomics, llms, transformer-models, antibiotic-resistance-prediction`  
  — LLMTB introduces a transformer-based large language model for predicting antibiotic resistance in *Mycobacterium tuberculosis* (MTB), trained on genomic data from 12,185 CRyPTIC isolates and evaluated on an independent set of 5,954 isolates. Motivated by the limitations of culture-based susceptibility testing and curated mutation-based tools (TBProfiler, Mykrobe, ResFinder, KvarQ), LLMTB leverages BERT-style architectures to extract genomic resistance signals directly from sequences without relying on predefined resistance variant lists. The model employs gene-level tokenization to capture biologically meaningful patterns and enhance interpretability. LLMTB achieves high predictive performance across 13 antibiotics—often matching or surpassing traditional AMR tools—while enabling fine-tuning and few-shot learning to rapidly adapt to emerging drugs. Attention analyses highlight relevant genes and intergenic regions, revealing known and potentially novel resistance determinants. Beyond accurate AMR classification, LLMTB offers deeper biological insights and demonstrates the potential of LLMs to generalize across genomic contexts, supporting improved diagnostics and personalized treatment strategies for drug-resistant TB.

- *"Large Language Models Facilitate the Generation of Electronic Health Record Phenotyping Algorithms"*, 2024  
  `ehr-phenotyping, clinical-informatics, llms, sql-generation, structured-health-data`  
  — This study evaluates the ability of four LLMs—GPT-4, GPT-3.5, Claude 2, and Bard—to generate computable electronic health record (EHR) phenotyping algorithms expressed as SQL queries aligned with a common data model (CDM). Phenotyping experts assessed LLM-generated algorithms for three clinical conditions (type 2 diabetes, dementia, and hypothyroidism) across instruction following, concept identification, logic accuracy, and executability. GPT-4 and GPT-3.5 outperformed Claude 2 and Bard, showing strong capabilities in identifying relevant clinical concepts such as diagnosis codes, medications, and lab measurements, and translating them into CDM-compatible SQL structures. However, both GPT models struggled with constructing logically coherent inclusion/exclusion criteria, often producing algorithms that were either overly restrictive (low recall) or overly permissive (low PPV). Implementation of the top-rated LLM-generated algorithms demonstrated that expert review and refinement remain necessary to achieve clinically valid phenotypes comparable to the eMERGE gold-standard algorithms. The study highlights LLMs’ potential to accelerate early-stage phenotyping—particularly literature synthesis and initial rule drafting—while emphasizing the continued need for clinical and informatics expertise to ensure reliability and reproducibility in EHR-based research.

- *"Advancing Plant Biology Through Deep Learning–Powered Natural Language Processing"*, 2024  
  `plant-biology, protein-language-models, dna-llms, deep-learning, agricultural-bioinformatics`  
  — This perspective highlights the growing impact of deep learning—particularly large language models (LLMs) and protein language models (PLMs)—on plant biology and agricultural genomics. The authors describe how transformer-based PLMs enable large-scale representation learning directly from DNA and protein sequences, capturing multiscale structural, functional, and evolutionary patterns that traditional computational methods fail to model. These models support tasks such as regulatory motif identification, protein structure–function inference, variant impact prediction, and trait-associated sequence analysis. The article emphasizes their relevance for accelerating crop improvement, enabling model-guided genome editing, and uncovering biological mechanisms underlying complex plant traits. By integrating LLM-powered analyses with existing plant omics datasets, researchers can enhance predictions of gene function, understand plant cell systems, and design strategies for sustainable agroecological transitions. The authors argue that deep learning, anchored by LLMs and PLMs, will play a central role in future plant sciences, bridging molecular biology with large-scale agricultural applications while requiring continued human oversight due to ethical, regulatory, and interpretability considerations.

- *"WEFormer: Classification for physiological time series with small sample sizes based on wavelet decomposition and time series foundation models"*, 2025  
  `physiological-time-series, tsfm, wavelet-decomposition, small-sample-learning, transformer-models`  
  — This study introduces WEFormer, a physiological time-series classification framework designed specifically for scenarios with extremely limited sample sizes, a common challenge in biomedical research due to high data-collection cost, privacy restrictions, and difficulty recruiting subjects. The model integrates a pretrained Time Series Foundation Model (TSFM; MOMENT) with frozen weights as a universal feature extractor, allowing the architecture to leverage rich temporal representations without overfitting. In parallel, WEFormer incorporates a differentiable MODWT wavelet decomposition module that separates input signals into multi-frequency sub-bands; a learnable attention mechanism dynamically emphasizes informative frequency components while suppressing noise, enabling robust feature learning even from low-quality wearable-device signals such as GSR and ECG. The approach avoids task-specific priors required by meta-learning or prototypical methods and instead combines generalized pretrained embeddings with adaptive spectral filtering. Extensive experiments on two multimodal datasets—WESAD for emotion recognition and MOCAS for cognitive workload estimation—show substantial accuracy improvements over prior deep-learning and augmentation-based methods under small-sample conditions. Ablation studies confirm that TSFM embeddings and wavelet-attention decomposition are both central to WEFormer’s performance, demonstrating the value of foundation time-series models for practical clinical and affective-computing applications with limited data.

- *"Semi-supervised learning with pseudo-labeling compares favorably with large language models for regulatory sequence prediction"*, 2025  
  `regulatory-genomics, semi-supervised-learning, pseudo-labeling, cross-species-alignment, dnabert2-benchmarking`  
  — This work proposes a cross-species semi-supervised learning (SSL) framework that substantially expands training data for regulatory sequence prediction by pseudo-labeling homologous regions across mammalian genomes, addressing the fundamental limitation that supervised deep learning relies on scarce functional genomic labels constrained by the finite size of the human genome. The method remaps annotated regulatory sequences (e.g., TF-binding peaks) from human to closely related genomes, generating large-scale pseudo-labeled datasets for model pretraining, followed by fine-tuning on the original labeled human data. An enhancement inspired by the Noisy Student algorithm estimates pseudo-label confidence and particularly improves performance for transcription factors with very small training sets. The SSL paradigm is architecture-agnostic and was applied to DeepBind, DeepSEA, and DNABERT2, consistently yielding stronger sequence-classification accuracy and improved SNP-effect prediction across multiple TFs and assays. Notably, compact SSL-trained models achieved performance matching or surpassing large DNA language models such as DNABERT2, highlighting that data-efficient SSL pretraining can rival or outperform computationally expensive self-supervised LLMs trained on many genomes. The study demonstrates that evolutionary conservation provides a powerful signal for regulatory model scaling without the heavy resource cost of large LLM pretraining.

- *"From text to traits: exploring the role of large language models in plant breeding"*, 2024  
  `plant-breeding, plant-omics, llms, multimodal-integration, genetic-relationship-mining`  
  — This review examines how large language models, originally designed for natural language understanding, can be repurposed as computational engines for uncovering complex genetic, phenotypic, and environmental interactions in modern plant breeding. The paper outlines how LLMs and foundational transformer architectures can ingest heterogeneous biological information—multi-omics, field phenotyping, environmental metadata, and genomic variation—to reveal non-linear genotype–phenotype relationships, infer novel genetic interactions, and improve trait-performance prediction. By treating biological sequences, trait descriptions, and environmental conditions as structured "text," LLMs can build unified latent representations that enhance selection strategies compared to traditional quantitative genetics approaches. The author highlights the potential of LLM-driven knowledge graph construction, multimodal data fusion, and zero-shot reasoning to support breeders in decision-making, accelerate discovery of beneficial alleles, and enable more sustainable crop-improvement pipelines. Despite these opportunities, the review emphasizes challenges including limited plant-specific training corpora, data heterogeneity, scarcity of labeled phenotypic datasets, and risks of hallucinations, ultimately framing LLMs as an emerging but still underexplored direction for advancing computational plant breeding.

- *"Large AI Models in Health Informatics: Applications, Challenges, and the Future"*, 2023, JBHI  
  `foundation-models, health-informatics, multimodal-health-data, medical-ai, llms`  
  — This comprehensive review outlines the rise of large AI models—foundation models trained on massive multimodal biomedical datasets—and their transformative impact on health informatics. The authors analyze how these models, exemplified by ChatGPT-scale architectures, reshape methodologies across seven major sectors: bioinformatics, medical diagnosis, medical imaging, EHR-based informatics, medical education, public health, and medical robotics. The review highlights how transformer-based large models integrate heterogeneous biological and clinical data (genomics, imaging, structured EHRs, medical text), enabling improved prediction, reasoning, and decision-support across health workflows. It also discusses architectural advances, self-supervised pretraining approaches, and cross-modal representations that underpin the scalability and adaptability of foundation models in healthcare. Despite their rapid progress, the paper emphasizes critical challenges—including robustness, bias, hallucinations, data scarcity for specialized diseases, privacy constraints, and computational cost—while offering forward-looking insights on safe deployment, regulatory considerations, and the role of trustworthy, clinically aligned large models in the future of health informatics.

- *"Deciphering enzymatic potential in metagenomic reads through DNA language models"*, 2024  
  `dna-llms, metagenomics, enzymatic-annotation, foundation-models, read-level-analysis`  
  — This study introduces two transformer-based DNA language models—REMME, a foundational model pretrained on raw metagenomic reads to learn contextual DNA representations, and REBEAN, a fine-tuned enzymatic-function annotator that predicts enzyme activities directly from unassembled metagenomic reads. Unlike traditional reference-based pipelines that depend on sequence homology, curated databases, or assembled contigs, REBEAN identifies functional signatures in short reads by focusing on molecular function rather than strict gene identity. The models demonstrate strong generalization to both known and orphan sequences, uncovering functionally relevant subsequences even without explicit supervision. Importantly, REBEAN expands enzymatic annotation coverage for environmental metagenomes and enables the discovery of previously uncharacterized enzymes, offering a reference-free strategy for probing microbial functional “dark matter”.

- *"Natural language processing data services for healthcare providers"*, 2024  
  `clinical-nlp, ehr-text-mining, snomed-ct, ner-pipelines, healthcare-integration`  
  — This paper presents a first-of-its-kind clinical NLP service deployed within the UK National Health Service (NHS), designed as an integrated data-processing and annotation framework to align machine learning workflows with real clinical environments. Using harmonised parallel platforms, the authors developed a scalable infrastructure for clinical text annotation, data quality management, and model refinement. The system distils expert clinician knowledge into NLP models through continuous annotation cycles, resulting in more than **26,086 manual annotations across 556 SNOMED-CT concepts**. The service primarily leverages named entity recognition (NER) for extracting diagnoses, procedures, and clinical attributes from unstructured EHR text, enabling downstream operational and clinical decision-support applications. By embedding NLP capabilities directly into provider workflows, the approach improves data accessibility, informs analytics, and supports broader adoption of AI-driven healthcare solutions. The authors argue that such vertically integrated NLP services will soon become standard components of healthcare delivery infrastructures.

- *"Deciphering genomic codes using advanced NLP techniques: a scoping review"*, 2024  
  `genomics-nlp, dna-tokenization, transformers, llms, regulatory-annotation, dna-language-models`  
  — This scoping review synthesizes recent efforts applying Natural Language Processing (NLP) and transformer-based Large Language Models (LLMs) to genomic sequencing data analysis. Surveying 26 studies published between 2021 and 2024, the paper highlights how DNA tokenization strategies (k-mers, adaptive tokenizers, byte-pair encodings), together with transformer architectures, improve the representation of genomic sequences by capturing long-range dependencies and regulatory patterns. The reviewed models demonstrate strong performance in predicting functional genomic annotations, including transcription-factor binding, chromatin accessibility, enhancer/promoter states, and variant-effect inference. The authors emphasize that transformer-based genomics models enable scalable processing of large sequencing data and facilitate regulatory code interpretation, yet challenges remain in transparency, dataset accessibility, model bias, and biological interpretability. The review positions genomic NLP as a rapidly emerging field with significant potential to advance precision medicine through automated, high-resolution analysis of noncoding regulatory regions.

- *"Review and reconciliation: A proof-of-concept investigation"*, 2025  
  `clinical-llms, medication-review, drug-safety, pharmacogenomics, decision-support`  
  — This proof-of-concept study evaluates the ability of four large language models—ChatGPT, Gemini, Claude-Instant, and Llama—to support medication review and reconciliation workflows. The authors assessed LLM performance across key pharmacotherapy tasks, including detection of dosing-regimen errors, identification of drug–drug interactions, therapeutic-drug-monitoring (TDM)–based dose adjustments, and genomics-guided individualized dosing. Outputs were evaluated using predefined criteria (accuracy, relevance, risk-management behavior, hallucination control, and citation quality). Results show variable but generally consistent model behavior: ChatGPT demonstrated high accuracy in most dosing-error scenarios; all LLMs correctly identified warfarin-related interactions but collectively missed the clinically important metoprolol–verapamil interaction. Claude-Instant provided the most appropriate recommendations for TDM-based regimen adjustment and pharmacogenomic decision-making, while Gemini was notable for spontaneously including citations and guideline references, enhancing interpretability. Error-impact analysis revealed minor safety implications for dosing-regimen and TDM tasks, but potentially major consequences for missed drug–drug interactions or incorrect pharmacogenomic recommendations. The study highlights both the promise and current limitations of LLMs as medication-review assistants and underscores the need for validated integration into EHR and prescribing systems to ensure safe deployment in clinical workflows.

- *"Conversational AI agent for precision oncology: AI-HOPE-WNT integrates clinical and genomic data to investigate WNT pathway dysregulation in colorectal cancer"*, 2025  
  `precision-oncology, llm-agents, wnt-pathway, colorectal-cancer, clinical-genomics-integration`  
  — This study introduces **AI-HOPE-WNT**, the first conversational AI agent dedicated to interrogating WNT-pathway dysregulation in colorectal cancer (CRC), with a particular focus on early-onset CRC and disparities across demographic groups. Built on a modular architecture that couples large language models with a natural language–to–code engine and automated statistical pipelines, the system interfaces directly with harmonized cBioPortal datasets. AI-HOPE-WNT enables mutation-frequency profiling, odds-ratio testing, survival analyses, subgroup stratification, co-mutation discovery, and treatment-response evaluation through natural language queries. Validation analyses showed the system accurately recapitulated findings from earlier studies, including elevated **RNF43** and **AXIN2** mutation frequencies in Hispanic/Latino populations and improved survival among WNT-altered EOCRC cases. Exploratory analyses generated novel insights such as survival differences associated with **APC** mutations in FOLFOX-treated EOCRC (p = 0.043), adverse outcomes of **RNF43** mutations in metastatic disease (p = 0.028), niche co-mutation enrichment (AXIN1–APC) across colon vs. rectal tumors, and demographic-MSI-specific effects of **AXIN2** mutations. The platform demonstrates how LLM-augmented analytics can democratize high-dimensional, pathway-focused oncology research and accelerate hypothesis generation. AI-HOPE-WNT is openly available at https://github.com/Velazquez-Villarreal-Lab/AI-HOPE-WNT.
### Tools, Software, and Databases Used
- **Large Language Models (LLMs):** Natural language–to–code generation engine (model family not explicitly named).  
- **Backend Analytics:** Automated statistical pipeline supporting mutation load analysis, odds ratios, survival modeling, and subgroup analysis.  
- **Databases:** Harmonized colorectal cancer genomics/clinical datasets accessed through **cBioPortal** (TCGA, GENIE, and associated CRC cohorts).  
- **Platform:** AI-HOPE-WNT conversational agent (open-source implementation on GitHub).
### Benchmarks, Evaluation Metrics, and Comparison Baselines
- **Benchmarking strategy:** Recapitulation of previously published WNT-pathway CRC studies involving demographic disparity analyses.  
- **Evaluation endpoints:**  
  - Reproduction of expected trends (e.g., RNF43/AXIN2 mutation rates; survival advantages in WNT-altered EOCRC).  
  - Statistical significance metrics: *p-values* from survival models and frequency comparisons.  
  - Consistency with prior ethnically stratified mutation findings.  
- **Baselines:**  
  - Prior manual analyses from WNT-focused CRC disparity studies.  
  - Established statistical results on APC, RNF43, AXIN2 mutation patterns in EOCRC vs. NHW populations.  
- **Exploratory benchmarks (no prior baseline):**  
  - Treatment-response associations (e.g., APC × FOLFOX).  
  - Tumor-location-specific co-mutation enrichment.  
  - Interaction of AXIN2 status with gender and MSI subtype in survival outcomes.

- *"Foundation models for generalist medical artificial intelligence"*, 2023, **Nature**  
  `foundation-models, generalist-medical-ai, multimodal-llms, self-supervised-learning, medical-reasoning`  
  — This perspective paper proposes **Generalist Medical AI (GMAI)** as a transformative paradigm for future clinical artificial intelligence. Unlike traditional task-specific medical models, GMAI systems are envisioned as *foundation models* trained via large-scale self-supervision across multimodal medical data—including imaging, EHRs, lab results, genomics, clinical text, and graphs. These models would dynamically adapt to new tasks through natural-language instructions (in-context learning), eliminating the need for task-specific fine-tuning. GMAI is expected to produce expressive outputs such as free-text radiology reports, reasoning chains, image annotations, and spoken clinical recommendations. The paper outlines high-impact applications (clinical decision support, triage, multimodal diagnostics, population health modeling), technical requirements (scalable multimodal architectures, contrastive pretraining, instruction tuning), and the massive training datasets needed. It also highlights emerging challenges in regulation, validation, medical data governance, and safety—especially as these systems blur the boundary between assistive tools and autonomous clinical reasoning agents.
### Tools, Software, and Databases Used
*(This is a conceptual framework paper rather than an empirical study; therefore no specific implementation tools or datasets are directly used. Instead, the paper references classes of technologies and architectures.)*  
- **Model families referenced:** GPT-3, Gato, multimodal transformers, contrastive models, early medical foundation models (e.g., BioGPT, Med-PaLM, RETFound).  
- **Algorithmic foundations:**  
  - *Self-supervised learning:* masked modeling, contrastive learning.  
  - *In-context learning* for dynamic task specification.  
  - *Multimodal fusion architectures* integrating images, text, EHR tables, and genomic data.  
- **Data modalities emphasized:** Imaging (X-ray, CT, MRI), EHR structured data, clinical text corpora, lab results, genomic sequencing data, graph-structured biomedical networks.
### Benchmarks, Evaluation Metrics, and Comparison Baselines
*(Again, this is a theoretical/position paper—no experiments or quantitative benchmarks were conducted. However, the authors describe the required evaluation strategies for future GMAI systems.)*  
- **Evaluation paradigms proposed:**  
  - Benchmark GMAI on *multimodal, multitask* performance across imaging, text, EHR, and genomics.  
  - Assess *generalization to unseen tasks* using in-context learning.  
  - Evaluate *clinical reasoning quality* via free-text explanations and structured outputs.  
  - Stress-test robustness across patient subgroups, institutions, and modalities.  
- **Comparison baselines:**  
  - Conventional task-specific medical models (e.g., pneumonia-only chest X-ray classifiers).  
  - Early medical foundation models for single modalities.  
  - General-purpose foundation models adapted to medicine (e.g., GPT-3, Gato) as conceptual reference points.

- **EDS-Kcr: Deep Supervision Based on Large Language Model for Identifying Protein Lysine Crotonylation Sites Across Multiple Species** (2024)  
  `protein-llms, esm2, ptm-prediction, crotonylation, multi-species, deep-supervision`
  - EDS-Kcr introduces a next-generation predictor for lysine crotonylation (Kcr) sites by integrating the protein large language model **ESM2** with a **deep supervision** architecture to enhance robustness and generalization across species. The model fuses classical sequence encodings (1-mer, 2-mer) with transformer-based embeddings to capture subtle biochemical patterns in protein sequences. Unlike prior Kcr predictors, EDS-Kcr supports **cross-species prediction**, covering human, plant, animal, and microbial proteins.  
  Through attention and visualization mechanisms, the framework provides interpretable biological insights into residues driving predictions. Across all benchmarks, EDS-Kcr consistently outperforms classical machine-learning models (e.g., CKSAAP_CrotSite, LightGBM-CroSite) and deep learning methods (Deep-Kcr, DeepCap-Kcr), achieving superior accuracy, AUC, and stability. This positions EDS-Kcr as a powerful tool for PTM annotation, disease research, and drug development.  
  **Webserver:** http://eds-kcr.lin-group.cn/
  **Tools / Models / Software Used**
  - Protein language model **ESM2** for deep contextual embeddings  
  - Deep supervision neural architecture  
  - Attention and visualization modules for interpretability  
  - 1-mer & 2-mer feature encodings  
  - Deep learning frameworks (e.g., PyTorch)
  **Datasets**
  - Multi-species lysine crotonylation (Kcr) datasets  
  - Human, plant, animal, and microbial protein sequences  
  - Benchmark PTM prediction datasets used for model evaluation  
  **Benchmarks, Metrics & Baselines**
  - **Metrics:** Accuracy, AUC/ROC, Sensitivity, Specificity, Precision, Recall, F1-score  
  - **Baselines Compared:**  
    - CKSAAP_CrotSite  
    - Position-weight Kcr predictor  
    - Deep-Kcr (CNN-based)  
    - LightGBM-CroSite  
    - DeepCap-Kcr (CNN + LSTM Capsule Network)  
  - **Outcome:** EDS-Kcr consistently surpasses all baselines in performance and cross-species generalization while providing improved interpretability.

- **Multiomics Research: Principles and Challenges in Integrated Analysis** (2024)  
  `multi-omics, integration, deep-learning, gnn, gan, llms-in-biology`
  - This review provides a comprehensive synthesis of the principles, computational frameworks, and challenges of multiomics research. It highlights how integrated analysis across genomics, transcriptomics, proteomics, metabolomics, and epigenomics enables deeper understanding of regulatory mechanisms and system-level biological processes. The authors outline major advances in machine learning techniques—including deep learning, graph neural networks (GNNs), and generative adversarial networks (GANs)—that enhance feature extraction, cross-modal data integration, and predictive modeling.  
  The review also emphasizes the emerging role of **large language models (LLMs)** in multiomics, particularly for automated feature extraction, natural-language-driven biological insight generation, knowledge integration, and hypothesis formation. The authors identify major challenges in multiomics workflows, such as data heterogeneity, scaling across modalities, interpretability, and the computational cost of training integrative models.  
  Collectively, this review provides a roadmap for next-generation multiomics research and highlights the synergy between high-throughput experimental technologies and advanced AI models.
  **Tools / Technologies / Models Discussed**
  - Deep learning architectures for multiomics integration  
  - Graph Neural Networks (GNNs) for modeling molecular interactions  
  - Generative Adversarial Networks (GANs) for data harmonization and augmentation  
  - Large Language Models (LLMs) for feature extraction and knowledge integration  
  - High-throughput sequencing platforms:  
    - Illumina NGS (short-read)  
    - PacBio Revio (HiFi long-read SMRT sequencing)  
    - Oxford Nanopore Technologies (real-time nanopore sequencing)  
  **Datasets / Data Modalities**
  - Genomic sequences (short-read FASTQ, long-read FASTQ/BAM)  
  - Transcriptomics (RNA-seq)  
  - Proteomics and metabolomics profiles  
  - Epigenomics: DNA methylation, chromatin accessibility  
  - Multiomics datasets integrating several modalities per sample  
  **Benchmarks, Metrics & Evaluation Baselines**
  - Evaluation dimensions emphasized in the review (not model-specific):  
    - Cross-modal predictive accuracy  
    - Data integration stability  
    - Scalability and computational efficiency  
    - Interpretability and biological consistency of integrated models  
  - Common baseline methods discussed:  
    - Classical statistical integration (e.g., PCA, CCA)  
    - Early/late fusion multiomics models  
    - Standard deep learning architectures versus GNN- and GAN-based integrators  
  - **Outcome:** AI-driven methods, especially deep learning and foundation models, show superior capacity to model complex cross-omics relationships but require significant computational resources and advanced model tuning.

- **Language Modelling Techniques for Analysing the Impact of Human Genetic Variation** (2024)  
  `variant-effect-prediction, llms, transformers, dna-language-models, protein-lms`
  - This systematic review analyses over fifty language-model-based approaches for predicting the functional impact of human genetic variants across DNA, RNA, and protein sequences. The review highlights the conceptual parallels between natural languages and biological sequences, motivating the use of NLP and transformer architectures for variant effect prediction. The authors outline how the introduction of transformers and large language models (LLMs) since 2017 has fundamentally reshaped the field by enabling long-range dependency modeling and improved capture of regulatory, structural, and evolutionary information.  
  - The paper systematically compares classical N-gram models, RNNs/LSTMs, CNN hybrids, transformer-based LMs, and post-transformer innovations, emphasizing their applications in predicting coding and non-coding variant effects, disease associations, protein function disruption, and regulatory impact. Despite the rapid progress, the review notes a lack of unified evaluation datasets and benchmarking frameworks, which currently limits reproducibility and cross-study comparison.  
  Overall, this review provides a comprehensive map of how language modelling paradigms—from small language models to large-scale transformers—have advanced computational variant interpretation and identifies emerging directions such as efficient small LMs, multimodal integration, and improved benchmarking resources.
  **Tools / Models Reviewed**
  - Classical NLP: N-grams, skip-grams, Word2Vec-style embeddings  
  - RNN-based models: LSTM, GRU  
  - CNN–RNN hybrids  
  - Transformer-based LMs: BERT, GPT-style models, DNABERT, RNABERT, ProtBERT, ESM models  
  - Post-transformer architectures and efficiency-oriented language models  
  - Evolutionary-scale models trained on multi-species data  
  **Datasets / Data Modalities**
  - Human variant datasets (coding & non-coding)  
  - Multi-species protein and DNA sequence corpora  
  - Variant effect assays: deep mutational scanning (DMS)  
  - Regulatory genomics datasets: TF binding, chromatin accessibility  
  - Clinical and population genetics resources (e.g., ClinVar, gnomAD)  
  **Benchmarks, Metrics & Baselines**
  - *Key finding:* No shared standardized benchmarking framework across studies  
  - Common evaluation metrics:  
    - AUROC, AUPRC  
    - Correlation with experimentally measured variant effects  
    - Classification accuracy for pathogenic vs. benign variants  
  - Frequently used baselines:  
    - Traditional ML models (SVMs, random forests)  
    - Evolutionary conservation scores (e.g., PSSM, phyloP)  
    - Earlier language models (LSTM-based, CNN-based)  
  **Outcome:** Transformer and LLM-based models consistently outperform classical and early deep learning baselines, especially for long-range regulatory variant interpretation, but the field urgently needs shared datasets and evaluation pipelines for fair comparison.

- **Multimodal Cell Maps as a Foundation for Structural and Functional Genomics** (2024)  
  `multimodal-genomics, protein-mapping, llm-annotation, structural-genomics, proteomics, imaging-integration`
  - This study presents one of the most comprehensive multimodal maps of human subcellular architecture to date, integrating biophysical protein–protein interaction profiles with immunofluorescence imaging for over 5,100 proteins in U2OS cells. Through self-supervised multimodal fusion, the authors resolve 275 distinct molecular assemblies spanning nanometre- to micrometre-scale organization, forming a hierarchical map of cellular structure. The map is systematically annotated using large language models (LLMs), enabling automated biological function inference across thousands of proteins and assemblies.
  - The authors validate the multimodal assemblies using an independent proteome-wide size-exclusion chromatography mass spectrometry (SEC–MS) dataset generated in the same cellular context. Key applications include deriving structural models for 111 heterodimeric complexes, expanding the structural understanding of Rag–Ragulator, identifying proteins with unexpected biological functions (e.g., C18orf21 in RNA processing, DPP9 in interferon signaling), and uncovering assemblies with multi-localization or cell-type-specific patterns. The map is further applied to pediatric cancer genomics, identifying 21 recurrently mutated assemblies and nominating 102 new cancer-associated proteins. All results are publicly accessible via the Cell Visualization Portal and Mapping Toolkit, establishing a foundational reference for future structural and functional genomics.
  **Tools, Software & Models**
  - Self-supervised multimodal integration pipeline (IF imaging + AP–MS)  
  - LLM-based functional annotation of assemblies  
  - SEC–MS (size-exclusion chromatography mass spectrometry) for validation  
  - AP–MS (affinity purification mass spectrometry)  
  - High-throughput immunofluorescence imaging  
  - Cell Visualization Portal & Mapping Toolkit  
  **Datasets**
  - 5,100 matched protein IF images and AP–MS interaction profiles (U2OS cells)  
  - Proteome-wide SEC–MS dataset (same cell line)  
  - Pediatric cancer genomic datasets (mutation recurrence analysis)  
  - Structural biology resources used for complex annotation  
  **Benchmarks, Evaluation Metrics & Baselines**
  - Validation via:
    - Concordance of multimodal clusters with SEC–MS complex elution profiles  
    - Comparison with known protein complexes and organellar markers  
    - Structural consistency checks for predicted heterodimeric complexes  
  - Baselines:
    - Previous 661-protein multimodal map  
    - AP–MS-only or IF-only clustering approaches  
  - Performance indicators include assembly coherence, cross-modality agreement, and structural plausibility scores  
  **Outcome:** The integrated multimodal map, enhanced by LLM-based annotation, serves as a high-resolution reference for human cell organization and an analytical framework for structural biology, protein function discovery, and cancer genomics.

- **The PRIDE database at 20 years: 2025 update** (2025)  
  `proteomics, mass-spectrometry, public-databases, FAIR-data, llm-tools, data-reuse`
  - This update provides a comprehensive overview of the evolution of the PRIDE database—currently the world’s largest public repository for mass spectrometry–based proteomics data and a core member of the ProteomeXchange consortium. Over the past three years, PRIDE Archive has scaled to an average of ~534 submitted datasets per month, driven by major infrastructure improvements including a high-performance Globus-based transfer system for very large datasets, an automated dataset validation pipeline, and a redesigned resubmission workflow.
  - Beyond infrastructural enhancements, PRIDE introduced several innovative features such as the PRIDE chatbot, built using open-source large language models (LLMs) to assist users with data discovery, submission guidance, and proteomics-related queries. The platform expanded support for complex workflows including MS crosslinking, top-down proteomics, immunopeptidomics, and DIA/DDA technologies. In parallel, PRIDE intensified efforts to systematically reanalyze and disseminate high-quality datasets into value-added biological resources such as UniProt, Ensembl, and Expression Atlas, strengthening integration between proteomics and multi-omics research.
  - As a founding ProteomeXchange member, PRIDE continues to lead development of interoperable open formats (e.g., mzML, mzIdentML, ProForma 2.0, SDRF-Proteomics, USI) through the PSI initiative, supporting FAIR data principles at scale. This 2025 update reinforces PRIDE’s position as a global core biodata resource, enabling reproducible proteomics, large-scale data reuse, and next-generation AI-driven analytics.
  **Tools, Software & Infrastructure**
  - PRIDE Archive (data deposition platform)  
  - Globus-based file transfer system for multi-TB datasets  
  - Automated dataset validation and integrity checking pipeline  
  - PRIDE Chatbot (LLM-powered assistant)  
  - PRIDE Resubmission & Curation pipelines  
  - PSI-supported open formats: mzML, mzIdentML, mzTab, ProForma 2.0, SDRF-Proteomics, Universal Spectrum Identifiers (USIs)  
  **Datasets**
  - >6,000 datasets/year (~534 per month) submitted to PRIDE Archive  
  - Full range of MS-based proteomics workflows:  
    - DDA, DIA, top-down, immunopeptidomics, crosslinking proteomics  
  - Reanalyzed datasets integrated into:  
    - UniProt, Ensembl, Expression Atlas  
  **Benchmarks, Evaluation Metrics & Baselines**
  - Dataset quality assessment via automated validation (format compliance, metadata completeness, spectral quality)  
  - Cross-resource consistency checks with ProteomeXchange standards  
  - Interoperability benchmarks using PSI formats (mzML, mzIdentML, SDRF)  
  - Baselines include:  
    - Previous PRIDE Archive submission rates and manual validation workflows  
    - Pre-chatbot querying tools  
    - Earlier ProteomeXchange submission schema  
  **Outcome:** PRIDE’s 20-year milestone marks a transition toward AI-supported, highly scalable proteomics data infrastructure, enabling more efficient data deposition, enhanced FAIR compliance, and deeper integration with genomic and transcriptomic resources.

- **ADAM-1: An AI Reasoning and Bioinformatics Model for Alzheimer’s Disease Detection and Microbiome–Clinical Data Integration** (2025)  
  `alzheimers-disease, llm, multi-agent-systems, microbiome, rag, clinical-data-integration`
  - ADAM-1 is a multi-agent reasoning framework built on large language models (LLMs) to integrate and interpret multimodal Alzheimer’s disease (AD) data, including gut microbiome profiles, clinical variables, and literature-derived biomedical knowledge. The system combines LLM-driven agents with retrieval-augmented generation (RAG) and modular chain-of-thought reasoning to enable contextual, explainable classification of AD in older adults. Unlike traditional single-modality approaches, ADAM-1 leverages coordinated agent roles for feature extraction, biological interpretation, and hypothesis grounding using external biomedical sources.
  - Evaluated on a curated dataset of 335 multi-omic clinical–microbiome samples, ADAM-1 considerably outperforms the XGBoost baseline, achieving significantly higher mean F1-score and notably lower variance—highlighting its robustness under data-limited conditions common in AD research. The model demonstrates strong stability when handling human-derived biological data and offers biologically grounded explanations by integrating immune, metabolic, and microbiome signatures implicated in AD pathogenesis. Future iterations aim to incorporate neuroimaging, peripheral biomarkers, and disease progression prediction, expanding ADAM-1 into a generalizable multi-omic reasoning platform for neurodegenerative disease research.
  **Tools, Software, and Framework Components**
  - Multi-agent LLM system for modular reasoning  
  - Chain-of-Thought (CoT) agents for interpretable inference  
  - Retrieval-Augmented Generation (RAG) pipelines for literature-based grounding  
  - LLM backbones referenced: GPT-4.5, OpenAI o1, Gemini 2.5, Claude Sonnet 3.7, DeepSeek R1  
  - Comparative baseline model: XGBoost classifier  
  - Data preprocessing modules for clinical and microbiome feature fusion  
  - Knowledge integration layer linking to biomedical literature and external ontologies  
  **Datasets**
  - Cohort: 335 samples from older adults  
  - Multimodal inputs:  
    - Gut microbiome profiles  
    - Clinical and demographic records  
  - Ground truth labels: Alzheimer’s disease vs. control (binary classification)  
  - Future data types planned:  
    - Neuroimaging, plasma biomarkers, peripheral immune markers  
  **Benchmarks, Evaluation Metrics, and Baselines**
  - Primary evaluation metric: Mean F1-score  
  - Variance analysis to assess prediction stability  
  - Baseline comparison: XGBoost (lower F1, higher variance)  
  - Model robustness assessed under data-limited settings  
  - Validation included biological plausibility checks using literature evidence via RAG  
  - No external benchmark dataset available (custom internal dataset)  
  **Outcome:** ADAM-1 demonstrates the utility of multi-agent LLM systems for multimodal biomedical reasoning, showing superior accuracy, stability, and interpretability compared to traditional ML models. It represents an early step toward scalable, generalist AI frameworks for neurodegenerative disease diagnostics.

- **DREAM: Autonomous Self-Evolving Research on Biomedical Data** (2025)  
  `autonomous-research, llm, agentic-ai, automated-bioinformatics, self-evolving-systems`
  - DREAM introduces the first fully autonomous, self-evolving biomedical research system capable of performing complete end-to-end scientific investigations without human intervention. Built around LLM-driven cognitive modules, DREAM autonomously generates scientific questions, interprets data, identifies relevant variables, configures computational environments, writes and corrects code, performs analyses, evaluates results, validates findings, and iteratively formulates deeper follow-up questions. Unlike prior semi-autonomous co-pilot systems (e.g., BIA, Bio-Copilot, DS-Agent, MLAgentBench, SciAgents), DREAM operates continuously and independently, scaling with computational resources and eliminating the need for manual environment setup or hypothesis formulation.
  - Validated in real-world biomedical settings, DREAM surpasses expert scientists in question-generation quality, achieves higher environment-setup success rates than experienced researchers, and discovers novel scientific insights. In its flagship evaluation on the Framingham Heart Study dataset, DREAM demonstrated more than a 10,000-fold efficiency improvement over human researchers, highlighting its transformative potential for large-scale biomedical discovery.
  **Tools, Software, and System Components**
  - Core LLM modules for:  
    - Autonomous scientific question generation  
    - Variable identification and task planning  
    - Code writing, debugging, and iterative refinement  
    - Data interpretation and hypothesis evolution  
  - Automated environment configuration engine  
  - Result evaluation and validation subsystem  
  - External Tool Library: **BMAP** (invoked within several modules)  
  - Integration with multi-agent reasoning paradigms and LLM orchestration frameworks  
  - Comparison references:  
    - Coscientist, ChemCrow (chemical automation)  
    - SciAgents (hypothesis generation)  
    - BIA, Bio-Copilot (bioinformatics assistants)  
    - ChatGPT ADA (clinical ML pipeline assistant)  
    - DS-Agent, MLAgentBench (ML development assistants)  
  **Datasets**
  - Real-world biomedical dataset:  
    - **Framingham Heart Study** (longitudinal cardiovascular cohort)  
  - DREAM operates as a general framework and is dataset-agnostic, but validation focuses on:  
    - Epidemiological variables  
    - Clinical risk factors  
    - Demographic and physiological measurements  
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**
  - Benchmarked against expert human researchers on:  
    - Question generation quality  
    - Environment configuration success rate  
    - Analytical completeness and correctness  
  - Performance metrics include:  
    - Success rate of autonomous environment setup  
    - Accuracy and validity of generated analyses  
    - Novel finding discovery rate  
    - Efficiency measured relative to human baseline (10,000× in Framingham study)  
  - No conventional ML benchmarks due to system-level evaluation  
  - DREAM outperformed:  
    - Top scientists (question generation)  
    - Experienced researchers (environment setup & task execution)  
  **Outcome:**  
  DREAM represents a paradigm shift toward fully autonomous, self-evolving scientific AI systems. By integrating LLM-driven reasoning, automated environment orchestration, and continuous hypothesis evolution, DREAM demonstrates the feasibility of scalable, 24/7 automated biomedical research pipelines, accelerating discovery far beyond human-only workflows.

- **Transformers and Genome Language Models** (2024)  
  `genome-language-models, transformers, gLMs, dna-llms, sequence-modeling`
  - This review outlines the emergence of genome language models (gLMs) built on transformer architectures, motivated by the analogy between human language and the genomic "code." It surveys how transformers and related architectures are reshaping genomic prediction tasks, including chromatin accessibility, transcription factor binding, 3D chromatin structure, regulatory element prediction, and variant impact interpretation. The authors highlight how gLMs trained using unsupervised objectives—masking, next-token prediction, or k-mer modeling—enable zero-shot and few-shot learning capabilities, bypassing the limitations of supervised genomic models that rely on assay-specific labeled data.
  - The review evaluates the strengths and limitations of transformers in genomics (notably attention scalability, long-range dependency modeling, and computational cost) and compares them to emerging alternatives such as state-space models (SSMs) that claim improved efficiency on long sequences. It also introduces hybrid architectures combining transformers with deep genomics predictors trained directly on assay-level outputs. The paper provides a roadmap for future genomic modeling, including efficiency improvements, multimodal integration, and architectures beyond standard transformers.
  **Tools, Architectures, and Model Families Discussed**
  - **Transformer-based gLMs**:  
    - DNABERT (and k-mer–based tokenization variants)  
    - Nucleotide Transformer  
    - HyenaDNA / MambaDNA (long-context transformer alternatives)  
    - Enformer (attention-based hybrid model for functional genomics)  
    - GenSLMs, Evoformer-inspired models  
    - Other masked or autoregressive genomic transformers used for chromatin and regulatory prediction  
  - **Alternative architectures beyond transformers**:  
    - **State-Space Models (SSMs)** for long DNA sequences  
    - Structured state-space models (S4)  
    - Mamba and related selective SSMs claimed to outperform transformers in specific long-range genomics tasks  
  - **Hybrid models for genomics** (transformers embedded inside task-specific networks):  
    - Models trained directly on high-dimensional assay data (e.g., chromatin accessibility, RNA-seq profiles)  
    - Assay-prediction networks using combined attention + convolutional modules  
  **Datasets**
  - The review summarizes datasets commonly used to train or validate gLMs, including:  
    - **Human and multi-species reference genomes** (HG38, mouse, yeast, plant genomes)  
    - Public regulatory genomics datasets:  
      - ENCODE (TF binding, DNase-seq, ATAC-seq)  
      - Roadmap Epigenomics (methylation, histone marks)  
      - Hi-C / Micro-C for chromatin structure  
    - Variant-impact benchmarking sets from:  
      - ClinVar  
      - gnomAD  
      - MPRA datasets for regulatory variant effects  
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**
  - Benchmarks discussed for evaluating gLMs include:  
    - **Regulatory element prediction metrics:**  
      - AUC, AUPRC, Pearson/Spearman correlation (for chromatin accessibility, TF binding)  
    - **Variant effect prediction benchmarks:**  
      - MPRA correlation scores  
      - Predictive accuracy on ClinVar pathogenicity distinctions  
    - **Sequence modeling efficiency metrics:**  
      - Long-range dependency capture  
      - Memory/computation scaling  
    - **Model family baselines:**  
      - CNN models (DeepSEA, Basset, Basenji)  
      - RNN-based models (LSTM, dilated RNNs)  
      - Transformer vs. SSM comparisons (Mamba, S4)  
      - gLMs vs. supervised task-specific models (e.g., Enformer)  
  **Outcome:**  
  The review positions genome language models as a rapidly advancing frontier in bioinformatics, demonstrating how transformers and emerging post-transformer architectures enable scalable, unsupervised modeling of genomic sequences. By enabling zero-/few-shot prediction and capturing long-range genomic interactions, gLMs offer a new paradigm for decoding regulatory logic, predicting functional consequences of variants, and moving toward foundation models for genomics.

- **AI-Empowered Perturbation Proteomics for Complex Biological Systems** (2024)  
  `perturbation-proteomics, foundation-models, systems-biology, deep-learning, pmmp-pipeline`
  - This perspective highlights the growing role of AI and foundation models in *perturbation proteomics*—a discipline centered on systematically perturbing biological systems and quantifying multilayer proteomic responses, including protein abundance, turnover, post-translational modifications, interactions, transport, and localization. The authors argue that limited availability of large-scale, high-quality perturbation datasets is a bottleneck for systems biology, and propose a unified framework (PMMP: **Perturbation → Measurement → Modeling → Prediction**) to standardize and scale perturbation-based biological discovery.
  - By integrating perturbation proteomics with machine learning and deep learning models, the field aims to infer mechanisms of action, uncover protein functions, optimize therapy selection, and guide compound or experimental design. The article draws analogies to physics-based approaches such as perturbation theory, percolation processes, and complex-systems modeling, illustrating how perturbation data reveal latent functional structure that reductionist methods miss. The authors envision building foundation models trained on large perturbation proteomic datasets—analogous to genomic or protein language models—to enable predictive, mechanistic modeling of biological response patterns.
  **Tools, Computational Methods, and Modeling Approaches**
  - **Machine Learning & Deep Learning for Perturbation Analysis**
    - Classical ML: regression, clustering, network modeling
    - Deep learning architectures for proteomic and systems data:
      - Graph Neural Networks (GNNs) for protein–protein interaction networks  
      - Foundation models for multi-scale perturbation response modeling  
      - Multi-task learning for generalizable perturbation predictions  
    - Causal inference and mechanism-of-action (MoA) prediction models  
  - **PMMP Pipeline Components**
    - *Perturbation*: biological, chemical, physical perturbations  
    - *Measurement*: MS-based proteomics (expression, turnover, PTMs, complexes) + phenotypes  
    - *Modeling*: AI/ML models integrating proteomics with protein networks  
    - *Prediction*: phenotype forecasting, drug response prediction, protein function inference  
  - **Conceptual Modeling Inspired by Physics**
    - Perturbation theory to approximate biological system behavior  
    - Percolation theory for network stability and critical transitions  
    - Complex-systems modeling to capture multi-layer proteomic responses  
    - Analogies to GraphCast (GNN-based global weather model) as a blueprint for biological forecasting  
  **Datasets**
  - Large-scale proteomic perturbation datasets (from chemical, genetic, and environmental perturbations)  
  - MS-based quantification layers:
    - Protein abundance and turnover datasets  
    - PTM datasets (phosphorylation, ubiquitination, etc.)  
    - Interaction proteomics (AP–MS, proximity labeling)  
    - Subcellular localization proteomics  
  - Phenotypic readouts linked to perturbations  
  - Integrative multi-omics perturbation datasets when available  
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**
  - **Perturbation response prediction metrics:**
    - Correlation (Pearson/Spearman) between predicted and measured proteomic shifts  
    - Accuracy of MoA classification  
    - Network-level robustness and perturbation propagation accuracy  
  - **Model comparison baselines:**
    - Traditional ML vs. deep learning  
    - Network-based models vs. transformer-like or foundation models  
    - Unperturbed vs. perturbed-system models  
  - **Evaluation dimensions:**
    - Generalization across perturbation types  
    - Transfer to new cell types or conditions  
    - Scalability and reproducibility of predictions  
  **Outcome:**  
  The article positions perturbation proteomics as essential for advancing systems biology and advocates for building large-scale perturbation-based foundation models. AI-backed PMMP pipelines can enable mechanistic predictions, enhance drug discovery, and reveal hidden protein functions, ultimately providing a high-throughput path toward comprehensive modeling of complex biological systems.

- **Deep Learning Applications Advance Plant Genomics Research** (2025)  
  `plant-genomics, deep-learning, gene-regulation, protein-prediction, plant-LLMs, multi-omics, transfer-learning`
  - This review provides a comprehensive synthesis of how deep learning (DL) has transformed modern plant genomics, driven by advances in high-throughput sequencing and the rapid expansion of plant multi-omics resources. The article covers DL applications across DNA, RNA, and protein sequence analysis, addressing gene regulatory element prediction, functional annotation, protein structure inference, and multi-omics integration in both model and horticultural crops. Neural architectures—including CNNs, RNNs, hybrid attention networks, graph neural networks, and transformer-based language models—have delivered substantial improvements in prediction accuracy, enabling discovery of regulatory mechanics, stress-response pathways, and evolutionary signals embedded in complex plant genomes. The review also highlights the growing ecosystem of plant-specific genomic language models such as **PDLLMs** and **AgroNT**, which leverage transformer pretraining on large-scale plant genomic corpora.
  **Tools, Computational Methods, and Model Architectures**
  - **Deep learning architectures in plant genomics**
    - *Convolutional Neural Networks (CNNs)* for motif discovery and regulatory element detection  
    - *Recurrent Neural Networks (RNNs / BiLSTMs)* for capturing long-range sequential dependencies  
    - *Transformers & Attention-based Models* for gene expression prediction, enhancer–promoter linking, and variant-effect modeling  
    - *Graph Neural Networks (GNNs)* for regulatory networks, gene–gene interactions, and chromatin topology  
    - *Protein structure models*: AlphaFold, ESM family for plant protein folding and functional inference  
  - **Plant-specific LLMs and genome language models**
    - **PDLLMs**: pretrained on diverse plant genome and transcriptome datasets  
    - **AgroNT**: nucleotide transformer optimized for agrigenomics tasks  
    - Applications include: regulatory annotation, variant interpretation, promoter classification, and trait-associated sequence prediction  
  - **Transfer learning strategies**
    - Cross-species genome transfer  
    - Multi-omics embedding transfer across tissues and developmental stages  
    - Fine-tuning LLMs for plant epigenomics, transcriptomics, and stress-response modeling  
  - **General DL workflow in plant genomics**
    - Data preprocessing → feature encoding → model architecture selection → training/validation → interpretation → biological validation  
  **Datasets**
  - Whole-genome sequences of major crops & horticultural plants  
  - ATAC-seq, ChIP-seq, RNA-seq, and methylation datasets for regulatory modeling  
  - Plant proteomes and structural datasets for DL-based folding and interaction prediction  
  - Functional genomics resources:
    - TAIR, Gramene, Phytozome, Ensembl Plants  
  - Trait- and phenotype-linked multi-omics datasets for stress resistance, development, and yield traits  
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**
  - **Metrics**
    - AUC, accuracy, F1-score for classification tasks  
    - MSE/MAE for expression and quantitative trait prediction  
    - Top-L precision for protein structure/contact prediction  
  - **Baselines**
    - Traditional ML: SVM, random forest, k-mer frequency classifiers  
    - Sequence-based vs. multi-omics-integrated DL models  
    - CNN/RNN architectures vs. transformer/LLM architectures  
    - Species-specific vs. cross-species transfer-learning models  
  - **Performance assessment dimensions**
    - Ability to capture long-range regulatory dependencies  
    - Cross-species generalization  
    - Data-efficiency and few-shot performance  
    - Interpretability and biological plausibility  
  **Outcome:**  
  The review demonstrates that DL—especially attention-based and LLM-based models—has significantly accelerated discovery in plant genomics, enabling deeper understanding of gene regulation, adaptive evolution, protein structure, and trait biology. However, progress remains constrained by limitations in high-quality plant annotations, computational cost, and the need for biologically interpretable models. The authors emphasize that future advances will depend on interdisciplinary collaboration to create scalable, transparent, and plant-optimized deep learning frameworks for next-generation genomic research.

- **Automatic Biomarker Discovery and Enrichment with BRAD** (2025)  
  `agentic-LLMs, RAG, biomarker-discovery, enrichment-analysis, reproducible-AI, biomedical-NLP`
  - This paper introduces **BRAD (Bioinformatics Retrieval Augmented Digital agent)**, an open-source, agentic AI system designed to automate biomarker discovery and literature-grounded enrichment interpretation. BRAD addresses major limitations of commercial LLM systems—such as lack of provenance, unverifiable outputs, and dependency on black-box APIs—by integrating transparent retrieval, modular tool orchestration, and reproducible data handling. The framework combines LLM reasoning with external databases, scientific literature, and bioinformatics tools to contextualize experimental findings (e.g., gene markers) within validated knowledge, thereby automating workflows that previously required extensive expert manual interpretation.
  **Biological and Computational Methodology**  
  - BRAD implements a modular agent-based architecture enabling LLMs to:  
    - Retrieve domain-specific knowledge from scientific literature  
    - Query bioinformatics databases and software pipelines  
    - Interpret biomarker lists and link them to biological pathways  
    - Generate reproducible enrichment analysis reports  
  - Uses **Retrieval Augmented Generation (RAG)** and **agentic decision-making** to reduce hallucinations and improve factual grounding.  
  - Provides transparent execution logs, explicit tool call history, and traceable references for every step—critical for biomedical reproducibility.  
  - Functions across heterogeneous workflows including biomarker identification, single-cell analysis, video RAG, and chatbot-style biomedical assistants.
  **Tools, Software, and Databases Used**  
  - **BRAD framework** (open-source Python package)  
  - External tool integration: literature APIs, database clients, and user-defined computational pipelines  
  - Commonly used bioinformatics resources (depending on workflow):  
    - Gene Ontology (GO)  
    - KEGG pathways  
    - PubMed / scientific literature via RAG  
    - Online biomarker or omics-specific repositories  
  - Modular agents allow attachment of custom analytic pipelines (e.g., expression analysis, clustering, enrichment engines)
  **Datasets**  
  - Demonstration case for biomarker discovery:  
    - Gene/marker lists supplied by the user (e.g., from differential expression, single-cell clusters, or proteomics data)  
    - Literature-derived biological knowledge retrieved programmatically  
  - BRAD is dataset-agnostic and can integrate:  
    - Bulk / scRNA-seq marker lists  
    - Proteomics biomarkers  
    - Multi-omics readouts  
    - Any structured or semi-structured biological data provided as input
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - Evaluation focused on **reproducibility, provenance, and interpretability**, addressing deficits found in commercial LLM systems.  
  - Key assessment dimensions:  
    - Transparency of tool chain  
    - Reduction of hallucinations via explicit RAG citations  
    - Reliability and verifiability of biomarker-to-literature mappings  
  - Comparisons made against:  
    - General-purpose chatbots (ChatGPT, Perplexity) which obscure data sources  
    - Existing RAG systems lacking biomedical-specific integration  
  - Outcome: BRAD provides more **auditable**, **consistent**, and **reproducible** results.
  **Results and Key Contributions**  
  - Fully automated biomarker discovery pipeline linking gene lists to functional literature.  
  - Automatic generation of enrichment analysis reports with source-verified citations.  
  - Achieves significantly higher reliability than commercial LLMs by exposing all tool interactions.  
  - Successfully deployed across diverse applications:  
    - Biomarker and enrichment workflows  
    - Clinical chatbots  
    - Video RAG systems  
    - Single-cell dataset analysis  
  - Highlights the importance of transparent, verifiable agentic systems in biomedical research.
  **Significance and Limitations**  
  - **Significance**:  
    - Establishes a reproducible, transparent AI system for automated biomarker interpretation.  
    - Bridges LLM reasoning with validated bioinformatics tools.  
    - Reduces manual workload and increases consistency of biological insight extraction.  
  - **Limitations**:  
    - Dependent on the coverage and quality of external databases.  
    - Requires integration effort for highly customized pipelines.  
    - Long-term performance depends on continuous updates to RAG sources and tool libraries.

- **Federated Deep Learning Enables Cancer Subtyping by Proteomics (ProCanFDL)** (2025)  
  `federated-learning, proteomics, cancer-subtyping, privacy-preserving-AI, multi-cohort, deep-learning`
  - This study introduces **ProCanFDL**, a federated deep learning (FDL) framework designed to enable large-scale cancer subtyping from clinically annotated proteomic data while preserving strict data-privacy constraints. The authors address a central limitation in biomedical AI—namely, the inability to assemble global datasets due to regulatory and ethical restrictions—by training local models on each institution’s private proteomics data, sharing only model parameters rather than raw samples. Across 40 tumor cohorts in eight countries, ProCanFDL demonstrates that high-performance histopathologic subtyping can be achieved without centralized data pooling, matching the accuracy of a conventional centralized model while significantly outperforming all locally trained baselines.
  **Biological and Computational Methodology**  
  - The proteomic data consist of **DIA-MS (data-independent acquisition)** and **TMT-MS (tandem mass tag)** technologies across multiple cancer types.  
  - Federated training procedure:  
    - Each institution trains a local deep neural network on its private proteomics matrix.  
    - Local weight updates are transferred to a secure aggregation server.  
    - A global model is iteratively updated without exposing any patient-level data.  
  - Evaluated on **14 cancer histopathologic subtypes**, later expanded to **16** using external datasets.  
  - Demonstrates federated learning as a viable approach for building multi-institutional proteomic AI systems under global privacy regulations (EU GDPR, local clinical storage policies, etc.).
  **Tools, Software, and Databases Used**  
  - **ProCanFDL custom federated deep learning framework**, including:  
    - Local deep learning classifiers for proteomics  
    - Secure aggregation server  
    - FL coordination protocols  
  - Proteomics pipelines (in cohorts):  
    - **DIA-MS**  
    - **TMT-MS**  
    - Standard MS analysis workflows for peak extraction and quantification  
  - No external omics databases are used in model training, as the work is dataset-centric and privacy-preserving.
  **Datasets**  
  - **ProCan Compendium (internal)**:  
    - *7,525 biospecimens*  
    - *30 cohorts*  
    - *19,930 DIA/MS runs*  
    - Samples preserved via both fresh-frozen and FFPE conditions  
  - **Simulated federated sites** for internal evaluation (n = 1,260 local, 6,265 remote samples).  
  - **Hold-out internal test set**:  
    - *625 samples*.  
  - **External validation datasets** (10 cohorts):  
    - *2 DIA-MS cohorts (n = 55)*  
    - *8 TMT-MS cohorts (n = 832)*  
    - Added two additional cancer subtypes, totaling **16** subtype classes.
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - Metrics used:  
    - Classification accuracy  
    - Performance gain relative to local models  
    - Cross-cohort generalizability  
    - Stability across technical MS platforms  
  - Baselines:  
    - **Local models** trained institution-by-institution  
    - **Centralized deep learning model** trained on fully merged data  
  - Performance highlights:  
    - **+43% improvement** over local-only training  
    - **Federated model ≈ centralized model performance**, despite not sharing data  
    - Maintains high accuracy across different MS technologies (DIA vs TMT)
  **Results and Key Contributions**  
  - First demonstration of real human cancer proteomics successfully modeled with federated deep learning.  
  - Achieved robust, high-accuracy subtype classification across **40 tumor cohorts in 8 countries**.  
  - Federated approach preserved full data privacy while enabling large-scale multi-institution model training.  
  - Demonstrated cross-technology generalization (DIA → TMT), critical for real-world proteomics variability.  
  - Provides a blueprint for **privacy-compliant international AI consortia** in proteomics, enabling biomarker discovery, drug-target identification, and future foundation-model development.
  **Significance**  
  - A practical solution for training global AI models when raw biomedical data cannot be shared.  
  - Enables proteomic-scale AI comparable to genomic federated initiatives (e.g., federated GWAS).  
  - Supports future **proteome foundation models**, with π-Hub cited as a target ecosystem.  
  - Accelerates cancer biomarker research by integrating heterogeneous cohorts without violating data governance laws.
  **Limitations**  
  - Model performance depends on consistent preprocessing pipelines across sites.  
  - Federated communication overhead may limit scalability for very large models.  
  - Some rare cancer subtypes had low representation, affecting classification stability.  
  - Does not yet incorporate multimodal data (genomics, histopathology images).  

- **Investigating the Prospects of ChatGPT in Training Medicinal Chemists and Novel Drug Development** (2025)  
  `llms, chatgpt, medicinal-chemistry, drug-discovery, chemoinformatics, nlp-tools`
  - This review examines the expanding role of ChatGPT and related large language models (LLMs) in medicinal chemistry, chemoinformatics, and computational drug discovery. The authors highlight how LLMs can accelerate several stages of the drug development pipeline by supporting scientific writing, code generation, dataset curation, structural similarity analysis, ADMET evaluation, virtual screening assistance, and educational training for new chemists. While emphasizing the efficiency gains offered by ChatGPT, the article also critically discusses the challenges of hallucinations, biases, reproducibility issues, and the inability of current models to process multimodal chemical data directly.
  **Biological and Computational Context**  
  - Medicinal chemistry integrates organic chemistry, structural biology, biophysics, pharmacology, and computational modeling.  
  - Early AI systems such as ELIZA and A.L.I.C.E laid the foundation for NLP-driven tools, culminating in modern LLMs like GPT-3, GPT-4, and PaLM.  
  - LLMs support computational drug design by improving code workflows, enabling natural-language scientific search, and assisting in in-silico modeling tasks relevant to structure-based and ligand-based drug development.
  **Tools, Software, and Databases Used**  
  The article is a *review* and does not introduce a new computational pipeline; however, it highlights key tools LLMs can interact with or accelerate:  
  - **ChatGPT (OpenAI)** — text generation, translation, code writing, documentation.  
  - **Chemoinformatics libraries** (mentioned as use-cases; e.g., RDKit, molecular similarity tools).  
  - **ML and DL frameworks** — for ADMET prediction and virtual screening (general references).  
  - **ChatGPT Plugins** (as demonstrated in cited work by Wang et al.) enabling:  
    - literature search  
    - similarity scoring  
    - ADMET evaluation  
    - virtual screening support  
  **Datasets**  
  *No new dataset introduced*, but the article discusses general categories that LLMs help analyze:  
  - Molecular structure datasets  
  - Drug-likeness and ADMET datasets  
  - Public chemical databases (implicit use-cases): PubChem, ChEMBL, DrugBank  
  - Training corpora for LLMs (unknown due to non-disclosure by developers)
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - The review does not perform empirical benchmarking.  
  - It highlights known limitations based on prior assessments:  
    - Hallucination rates  
    - Inability to verify citations  
    - Lack of multimodal input support  
    - Biases inherited from training data  
  - Compares **LLMs vs classical chemoinformatics tools** qualitatively (not quantitatively):  
    - LLMs useful for coding, reasoning, summarization  
    - Classical tools superior for numerical precision, ADMET modeling, docking accuracy  
    - Hybrid workflows are recommended  
  **Key Results and Insights**  
  - LLMs provide significant advantages in medicinal chemistry by:  
    - Accelerating literature synthesis and hypothesis generation  
    - Assisting in script generation for docking, QSAR, data preprocessing  
    - Supporting dataset cleaning and error detection  
    - Improving accessibility via multilingual translation  
    - Aiding education for trainee medicinal chemists  
  - Example use-case: ChatGPT plugins were used for cocaine addiction drug research to analyze similarity indices and ADMET properties of candidate molecules.
  **Significance**  
  - LLMs democratize access to computational drug design workflows.  
  - They help reduce barriers for early-career chemists by:  
    - Automating tedious coding tasks  
    - Providing rapid conceptual explanations  
    - Enhancing productivity in literature-based analyses  
  - They also highlight the potential for future AI-augmented medicinal chemistry environments where generative models assist in molecular ideation, retrosynthesis, and rapid annotation.
  **Limitations**  
  - High hallucination frequency, particularly in chemical property prediction and citation generation.  
  - Inability to process multimodal data (chemical structures, 3D conformers, spectra).  
  - Lack of transparency regarding training datasets.  
  - Risk of propagating social and scientific biases.  
  - Ethical issues: misinformation, job displacement, and misuse in generating harmful chemical knowledge.  
  - LLM outputs require expert verification before use in real drug-development pipelines.

- **Integrating Multimodal Cancer Data using Deep Latent Variable Path Modelling (DLVPM)** (2025)  
  `multimodal-integration, deep-learning, cancer-genomics, histopathology, path-modelling, tcga`
  - This study introduces **Deep Latent Variable Path Modelling (DLVPM)**, a new framework that integrates deep learning with structural path modelling to map complex dependencies across multimodal cancer datasets. Unlike traditional path-modelling methods that cannot handle unstructured data (e.g., histology images) or nonlinear biological interactions, DLVPM uses deep latent encoders to jointly model relationships among genomic, epigenomic, transcriptomic, microRNA, and histopathological features. Using the TCGA breast cancer cohort, the method demonstrated significantly better performance than classical path modelling in capturing cross-modal associations, and was further validated on single-cell RNA-seq, CRISPR–Cas9 gene dependency screens, and spatial transcriptomics.
  **Biological and Computational Background**  
  - Cancer integrates alterations across multiple biological layers: SNVs, methylation, miRNA regulation, transcription, and tissue morphology.  
  - Conventional multiomic integration struggles to infer **directional dependencies** and **latent interactions** among diverse modalities.  
  - Path modelling (structural equation modelling) can theoretically map these dependencies, but classical models cannot process high-dimensional or unstructured data.  
  - DLVPM bridges this gap by combining:  
    - deep learning encoders for complex data  
    - latent variable path structures for causal/statistical dependency mapping  
  **Tools, Software, and Databases Used**  
  - **DLVPM (proposed framework)** — deep-learning-based latent path modelling  
  - **Deep neural encoders** for:  
    - single-nucleotide variants (SNV)  
    - DNA methylation  
    - microRNA sequencing  
    - RNA-seq gene expression  
    - Histopathology whole-slide images  
  - **Comparison models included:**  
    - classical structural equation modelling (SEM)  
    - state-of-the-art histology deep learning models (CNN-based; unspecified but standard baselines)  
  - **External Tools for downstream validation:**  
    - CRISPR–Cas9 dependency datasets  
    - Spatial transcriptomics integration frameworks  
  - **Visualization & interpretation:**  
    - Path diagrams for multimodal dependency mapping  
    - Latent embedding visualization for subtyping  
  **Datasets**  
  - **TCGA Breast Cancer (BRCA)** — main multimodal dataset including:  
    - SNVs  
    - DNA methylation arrays  
    - miRNA-seq  
    - RNA-seq  
    - H&E histology images  
  - **Independent validation datasets:**  
    - Single-cell RNA-seq datasets (for subtyping)  
    - CRISPR–Cas9 knockout gene dependency screens (cell lines)  
    - Spatial transcriptomics datasets (for histology–transcriptional consistency)  
  **Benchmarks, Evaluation Metrics, and Baselines**  
  - **Performance comparison:**  
    - DLVPM vs **classical path modelling**:  
      - DLVPM showed superior capability in mapping cross-modal associations and nonlinear interactions.  
    - For histology processing:  
      - Benchmarked against state-of-the-art deep learning image models (CNN-based).  
  - **Metrics used (implicit / reported):**  
    - Correlation between latent variables  
    - Reconstruction loss of latent encoders  
    - Association strength between modalities  
    - Statistical significance of gene–histology associations  
  **Key Results**  
  - **1. Enhanced Multimodal Dependency Mapping**  
    - DLVPM identified complex relationships linking mutations → methylation → miRNA → gene expression → histologic phenotypes.  
    - Outperformed classical SEM in accuracy and stability.
  - **2. Discovery of Genetic Loci Associated with Histologic Features**  
    - Hundreds of SNV and epigenetic loci showed significant associations with tissue morphology.  
    - Indicates strong molecular–morphological integration.
  - **3. Generalization Across Datasets**  
    - Molecular subcomponents trained on TCGA generalized to:  
      - patient cohorts  
      - cancer cell lines  
      - CRISPR dependency datasets  
      - spatial transcriptomics samples  
  - **4. Synthetic Lethal Interaction Insights**  
    - Latent-space relationships identified gene dependencies correlating with CRISPR–Cas9 knockout sensitivities.  
  - **5. Single-Cell Subtype Stratification**  
    - Encoded latent variables successfully distinguished cell types and tumor subpopulations.  
  **Significance**  
  - Provides a **holistic model of cancer** integrating genetics, epigenetics, transcription, and histological structure.  
  - Bridges the gap between **deep learning** and **causal/statistical modelling**.  
  - Enables multiomic mechanistic exploration and new hypothesis generation in cancer genomics.  
  - Demonstrates that multimodal latent-space models can generalize across datasets and modalities.
  **Limitations**  
  - Requires large, well-annotated multimodal datasets (not widely available).  
  - Computational cost is high due to multiple deep encoders.  
  - Path interpretation remains partially dependent on latent-space assumptions.  
  - Not fully causal—identifies statistical dependencies but cannot guarantee directionality.  
  - Integration with clinical data (survival, therapy response) remains to be developed.

- **Deep Learning–Based Multimodal Biomedical Data Fusion: An Overview and Comparative Review** (2025)  
  `multimodal-data, deep-learning, biomedical-fusion, large-models, AIGC, data-integration`
  - This survey provides a comprehensive overview of **deep learning–driven multimodal biomedical data fusion**, emphasizing how integrating diverse data modalities (e.g., imaging, genomics, clinical text, physiological signals) can overcome the limitations of single-modality analysis and significantly improve diagnostic and predictive performance. The review categorizes multimodal fusion approaches into **data-level, feature-level, and decision-level fusion**, evaluates state-of-the-art deep learning architectures—CNNs, RNNs, attention mechanisms, and graph neural networks—and discusses the impact of emerging technologies such as **generative models and large language models** on biomedical fusion research. The authors also outline challenges including lack of interpretability, modality-specific architectures, scalability, and the absence of universal fusion frameworks.
  **Biological and Computational Background**  
  - Biomedical systems generate **heterogeneous modalities**: imaging, omics (genomics, transcriptomics), EHRs, wearables, biosignals, and human–computer interaction data.  
  - Each modality differs in structure, statistical distribution, sampling density, and biological meaning.  
  - Multimodal fusion leverages **complementary biological information**, improving disease diagnosis, subtype stratification, biomarker discovery, and precision healthcare.  
  - Deep learning enables **automatic representation learning**, capturing nonlinear, cross-modal dependencies that classical methods cannot model.  
  - Recent advances — attention, transformers, GNNs, LLMs — further expand the capability to unify structured and unstructured modalities.
  **Tools, Software, and Architectures Discussed**  
  *(Note: Being a survey, the paper reviews methods rather than proposing a new tool.)*  
  **Deep learning architectures reviewed:**  
  - Convolutional Neural Networks (CNNs): image fusion, pathology imaging, radiomics  
  - Recurrent Neural Networks (RNNs), LSTMs, GRUs: physiological signals, time-series biomedical data  
  - Attention mechanisms & Transformers: cross-modal alignment, multimodal embedding spaces  
  - Graph Neural Networks (GNNs): integrating biological networks, omics graphs, patient similarity graphs  
  - Autoencoders & Deep Generative Models (VAEs, GANs): representation learning, modality reconstruction  
  - Large Language Models (LLMs): clinical text integration, multimodal embeddings, AIGC-based knowledge integration  
  **Other tools and frameworks mentioned across the reviewed literature:**  
  - Fusion modules: concatenation, cross-attention, co-attention, tensor fusion networks  
  - Optimization/training approaches: multi-task learning, contrastive learning, self-supervised pretraining  
  - Data preprocessing pipelines for imaging, omics, EHR, signal data  
  **Datasets Covered in the Survey**  
  *(Curated list of representative multimodal biomedical datasets used in the field)*  
  - **Medical imaging datasets:**  
    - MRI, CT, PET datasets for neurological and oncological tasks  
  - **Genomics & multiomics datasets:**  
    - TCGA (multi-cancer genomics + imaging + clinical)  
    - GEO-based gene expression datasets  
    - Physiological signal datasets (EEG, ECG, EMG)  
  - **Clinical records & EHR datasets:**  
    - MIMIC-III / MIMIC-IV (clinical notes + vitals + labs + signals)  
  - **Multimodal integration datasets:**  
    - Radiogenomics datasets linking imaging to genomics  
    - Multisensor wearable datasets  
  - The survey highlights how these datasets serve as “cornerstones” for fusion research.
  **Benchmarks, Baselines, and Evaluation Metrics**  
  *(Survey summarizing benchmarking practices across modalities)*  
  - **Fusion-level baselines:**  
    - Data-level (early fusion)  
    - Feature-level (intermediate fusion)  
    - Decision-level (late fusion)  
  - **Typical evaluation metrics:**  
    - Classification: accuracy, F1, recall, precision, AUROC  
    - Regression: RMSE, MAE, R²  
    - Representation quality: contrastive loss, clustering scores, embedding separability  
    - Cross-modal reconstruction & consistency metrics  
  - **Baselines compared across literature:**  
    - Single-modality deep learning models  
    - Classical ML models (SVM, RF)  
    - Hybrid multimodal fusion pipelines  
  **Key Insights and Results from the Survey**  
  - **Deep learning dramatically improves multimodal fusion performance**, especially where modalities are high-dimensional and heterogeneous.  
  - **Feature-level fusion** generally outperforms data-level and decision-level approaches by capturing cross-modal interactions.  
  - Attention and transformer architectures provide **superior cross-modal alignment**, especially for clinical text + imaging + genomics integrations.  
  - GNNs effectively integrate **biological networks, pathway structure, and patient similarity graphs**, improving biomarker discovery.  
  - Large models (LLMs and multimodal AIGC) enable:  
    - automatic feature extraction  
    - cross-modality semantic alignment  
    - enhanced interpretability and contextual reasoning  
  **Significance**  
  - Establishes a unified taxonomy of multimodal fusion strategies.  
  - Highlights how deep learning enables **holistic patient modeling across modalities**, critical for precision medicine.  
  - Shows the transformative potential of LLMs and generative models in biomedical integration.  
  - Provides guidance on selecting fusion strategies based on data type, dimensionality, and clinical task.  
  - Serves as a reference roadmap for developing next-generation multimodal fusion architectures in healthcare.
  **Limitations and Open Challenges**  
  - Lack of **interpretability** across many deep fusion architectures.  
  - High computational cost and need for large annotated multimodal datasets.  
  - Models are often **modality-specific**, limiting generalizability.  
  - No universal fusion framework exists that works robustly across heterogeneous biomedical modalities.  
  - Data heterogeneity, missing modalities, and domain shifts remain unresolved.  
  - Privacy, reproducibility, and AIGC-generated data pose additional theoretical challenges.  
  - Integration of LLMs into certified clinical pipelines still faces regulatory and safety constraints.

- **AskBeacon — Performing Genomic Data Exchange and Analytics with Natural Language** (2025)  
  `genome-analysis, beacon-protocol, llm-agents, secure-genomics, natural-language-queries, ga4gh`
  - AskBeacon introduces a natural-language interface that enables clinicians and researchers to query globally distributed genomic datasets using the **GA4GH Beacon protocol** without requiring programming expertise. The system uses large language models (LLMs) to translate user questions into Beacon-compliant genomic queries, execute secure data retrieval, and automatically generate publication-ready analyses and visualizations. Using the Parkinson’s Progression Markers Initiative (PPMI) cohort, the authors demonstrate AskBeacon’s ability to detect sex-based differences in autosomal versus X-linked variants. The platform integrates strict safety guardrails to prevent genomic data leakage, sanitize generated code, and mitigate hallucinations—making it suitable for secure genomic analytics across federated cohorts.
  **Biological and Computational Background**  
  - Global genomic cohorts (e.g., PPMI, gnomAD-derived datasets) are increasingly accessed via **federated search protocols** such as GA4GH Beacon.  
  - Beacon servers respond to presence/absence queries about variants without sharing raw data, supporting privacy-preserving genomics.  
  - AskBeacon extends Beacon by allowing **natural-language interaction**, removing computational barriers for clinicians.  
  - LLMs act as intermediaries that generate valid queries, post-process Beacon responses, analyze allele frequencies, and produce interpretable visual summaries.  
  - This bridges clinical genetics with secure, federated computational genomics.
  **Tools, Software, and Databases Used**  
  - **GA4GH Beacon Protocol**: Standard for federated genomic variant presence queries.  
  - **PPMI Dataset**: Parkinson’s Progression Markers Initiative cohort used as a demonstration dataset.  
  - **LLM backends evaluated**:  
    - Commercial: GPT family  
    - Open-weight: Llama variants, other open-source LLMs  
  - **AskBeacon components**:  
    - Natural-language → Beacon query translation engine  
    - Code generation module (sanitized R/Python)  
    - Execution sandbox to prevent data leakage  
    - Visualization pipeline for generating publication-ready figures  
  - **Programming tools**:  
    - Secure execution environment for LLM-generated analysis code  
    - Internal safety module enforcing policy constraints (hallucination mitigation, query validation)
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - **Models compared**: commercial vs open-weight LLMs, transformer architectures  
  - **Evaluation criteria**:  
    - Query translation accuracy  
    - Hallucination rate and correctness of generated code  
    - Safety performance (data exposure risk)  
    - Reproducibility of analyses  
    - Visualization quality (publication-ready standard)  
  - **Baselines**:  
    - Standard Beacon command-line queries  
    - Manual scripting-based genomic querying pipelines  
  - AskBeacon surpasses baselines by enabling **end-to-end automated analysis**, not just variant lookup.
  **Key Findings and Results**  
  - AskBeacon resolves the PPMI case study by determining:  
    - The autosomal marker occurs **1.4× more frequently** in males with Parkinson’s disease than females.  
    - The tested X-linked marker shows **no significant sex difference**.  
  - Automatically generates scientific figures and statistical summaries without user coding.  
  - LLMs can reliably translate natural-language clinical questions into federated genomic queries when safeguarded against hallucination.  
  - Secure architecture prevents direct LLM access to genomic data, ensuring compliance with privacy regulations.
  **Significance**  
  - Provides a **clinically accessible**, LLM-powered gateway to global federated genomic datasets.  
  - Accelerates genomics research by democratizing Beacon querying and analysis.  
  - Reduces technical burden on clinicians, enabling faster insights into population-level genetics.  
  - Advances secure, explainable LLM use in genomics by combining **privacy-preserving protocols + controlled code generation**.  
  - Establishes a blueprint for future natural-language interfaces to federated multiomics systems.
  **Limitations and Challenges**  
  - Dependence on Beacon schema restricts the complexity of supported genomic queries.  
  - LLM performance varies across models; some require extensive guardrails to prevent unsafe output.  
  - Lack of direct access to raw variant-level data limits downstream statistical analyses.  
  - Natural-language interpretation remains sensitive to ambiguous phrasing.  
  - Future extensions needed for structural variants, multiomic integration, and clinical metadata retrieval.

- **Genotypic–Phenotypic Landscape Computation Based on First Principle and Deep Learning** (2025)  
  `genotype-phenotype-mapping, fitness-landscape, transformer-models, viral-evolution, immune-escape, sars-cov-2, foundation-models`
  - This study establishes a theoretical and computational framework for quantitatively linking genotypes to phenotypes using a deep learning–based first-principle approach. The authors introduce the **Phenotypic-Embedding (P-E) theorem**, which formalizes the genotype→phenotype relationship within an encoder–decoder architecture, enabling the construction of **computable, interpretable genotype–fitness landscapes**. Built on this theorem, they develop a **Co-attention Transformer foundation model** that captures epistasis, recombination effects, immune escape, and viral transmissibility (R₀) directly from SARS-CoV-2 genomic sequences. The model accurately simulates neutral evolution, predicts immune escape mutations, and mathematically derives the viral basic reproduction number from sequence data alone. This framework provides a new paradigm for theoretical and computational biology by unifying first principles with deep learning for genotype–phenotype mapping.
  **Biological and Computational Background**  
  - Viral evolution is governed by genotype–phenotype interactions, where fitness depends on mutation effects, epistasis, and immune escape.  
  - SARS-CoV-2 offers a unique large-scale dataset (15M+ sequences from GISAID) enabling quantitative evolutionary modelling.  
  - Existing models (e.g., language-model viral embeddings, PyR0, statistical lineage-fitness models) either lack mechanistic interpretability or cannot account for higher-order epistasis and recombination.  
  - The P-E theorem bridges population genetics with sequence-based latent representations, enabling fitness to be treated as an expectation of hidden state variables.  
  - The Co-attention Transformer provides a scalable foundation model capable of representing non-linear genotype–phenotype mappings and generative prediction of immune escape.
  **Tools, Software, and Databases Used**  
  - **Datasets**:  
    - *GISAID* SARS-CoV-2 whole-genome sequences (15M+ entries with spatio-temporal metadata).  
  - **Models and Algorithms**:  
    - Encoder–Decoder Seq2Seq deep learning framework (theoretical basis of the P-E theorem).  
    - **Co-attention Transformer**, pretrained + SFT for fitness prediction.  
    - Latent variable modelling to express fitness as expected value in embedding space.  
  - **Software stack** (as implied):  
    - Python deep learning ecosystems (likely PyTorch/TensorFlow for Transformer implementation).  
    - High-performance computing infrastructure for large-scale sequence modelling.  
  - **Comparative Baselines**:  
    - Natural-language viral sequence models (Hie et al.).  
    - PyR0 Bayesian lineage-fitness regression (Obermeyer et al.).  
    - Epidemiological covariate-integrated statistical models (Maher et al.).
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - **Baselines used for comparison**:  
    - LM-based viral embeddings for mutation effect prediction.  
    - Bayesian & statistical fitness estimation models (PyR0, logistic regression, covariate-driven regressors).  
  - **Evaluation metrics**:  
    - Accuracy of neutral evolution simulation.  
    - Predictive accuracy of immune escape mutations.  
    - Correlation between predicted and empirical viral growth advantage / lineage fitness.  
    - Consistency and stability of R₀ estimation.  
  - **Results vs baselines**:  
    - More accurate modelling of epistasis and recombination effects.  
    - Better prospective and retrospective prediction of immune escape mutations.  
    - Increased interpretability through computable latent-variable–based fitness definition.  
    - Stronger capacity to construct high-resolution genotype–fitness landscapes.
  **Key Findings and Results**  
  - Formalization of the **Phenotypic-Embedding theorem**, providing a computable mapping between genotype sequences and phenotype variables.  
  - Development of a **first-principle Genotype–Phenotype framework**, suitable for landscape computation across biological systems.  
  - A **Transformer-based foundation model** that:  
    - Predicts immune escape mutations with high accuracy.  
    - Simulates neutral viral evolution trajectories.  
    - Computes viral fitness directly from genotype without epidemiological priors.  
  - Accurate first-principle derivation of SARS-CoV-2 **basic reproduction number R₀** from genomic embeddings.  
  - Construction of a mathematically grounded genotype–fitness landscape in latent embedding space.  
  - Model functions as a **generative engine** predicting future immune escape candidates, validated retrospectively and prospectively.
  **Significance**  
  - Establishes a **general, interpretable, first-principle framework** for genotype–phenotype landscape modelling.  
  - Provides the first demonstration that viral transmissibility (R₀) can be computed directly from sequence data.  
  - Captures complex evolutionary phenomena—epistasis, recombination, immune escape—using a unified Transformer model.  
  - Enables predictive surveillance of viral evolution and immune escape pathways.  
  - Unifies theoretical biology with foundation-model architectures, offering a reproducible and extensible paradigm for evolutionary modelling.
  **Limitations and Challenges**  
  - Requires extremely large and high-quality sequence datasets; performance may drop in low-data settings.  
  - Interpretation of latent variables, although theoretically grounded, may still rely on approximations of evolutionary dynamics.  
  - Real-world viral fitness is influenced by environmental/epidemiological factors not fully captured by genomic sequences alone.  
  - Generalization to organisms with larger genomes or lower mutation rates may require substantial architecture scaling.  
  - Computational cost of pretraining and modelling epistasis at large scale remains high.

- **Multimodal Integration Strategies for Clinical Application in Oncology** (2025)  
  `multimodal-integration, oncology, deep-learning, clinical-ai, multi-omics, EHR, imaging, vision-language-models, prognosis, biomarkers`
  - This review examines current artificial intelligence strategies for integrating heterogeneous multimodal data in oncology—spanning clinical records, bulk and single-cell multi-omics, spatial omics, radiology, histopathology, and wearable sensor data. The authors discuss the technical foundations of single-modality preprocessing (omics harmonization, imaging pipelines, NLP of clinical notes), followed by a synthesis of multimodal fusion strategies across machine learning, representation learning, and vision–language models. The review evaluates how multimodal fusion supports early cancer detection, biomarker discovery, prognosis prediction, and treatment response modelling, while analyzing limitations related to data heterogeneity, missingness, harmonization, computational demands, and lack of standardized multimodal workflows.
  **Biological and Clinical Background**  
  - Oncology now routinely generates high-resolution, multi-scale data: genomics, transcriptomics, proteomics, metabolomics, bulk + single-cell + spatial modalities, radiology (MRI, CT), histopathology, EHRs, and sensor data.  
  - Each modality captures different biological layers—mutational drivers, transcriptional programs, tumour microenvironment, tissue morphology, and patient-level clinical trajectories.  
  - Clinical decision-making still relies mostly on single-modality analyses, which fail to capture the full tumour complexity.  
  - Multimodal fusion allows orthogonal biological information to be combined, improving diagnosis, prognosis, and treatment-response prediction.  
  - Rapid growth of AI—especially deep learning and foundation models—enables scalable integration of complex, heterogeneous clinical data.
  **Tools, Software, and Databases Used (as discussed in the review)**  
  *Since this is a review, tools are referenced rather than implemented directly.*  
  - **Single-modality processing tools:**  
    - *Omics*: bulk RNA-seq preprocessing pipelines, single-cell analysis frameworks (Seurat, Scanpy), spatial transcriptomics alignment tools.  
    - *Imaging*: CNN-based pipelines, digital pathology WSIs, radiology preprocessing platforms.  
    - *Clinical text*: NLP models including BERT variants, ClinicalBERT, GPT-based LLMs for EHR structuring.  
  - **Multimodal model families:**  
    - **Traditional ML**: random forests, SVMs, elastic net, canonical correlation analysis (CCA).  
    - **Representation learning**: variational autoencoders (VAEs), graph neural networks (GNNs), contrastive learning frameworks.  
    - **Vision–language models (VLMs)** for pathology + text, radiology reports + images, clinical notes + biomarkers.  
    - **Deep multimodal transformers** for paired imaging–omics or imaging–EHR integration.  
  - **Clinical data sources referenced:**  
    - TCGA, CPTAC, UK Biobank, institution-level EHR systems, radiology archives, and emerging spatial-omics atlases.
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  (Summarized based on integration strategies surveyed in the review.)  
  - **Common evaluation metrics:**  
    - AUROC, AUPRC, accuracy, F1-score (diagnosis/prognosis tasks).  
    - C-index for survival prediction.  
    - Concordance for biomarker associations.  
    - Calibration curves for clinical outcome models.  
  - **Benchmarks referenced across literature:**  
    - *Multimodal fusion benchmarks:* TCGA multimodal tasks, multimodal pathology-to-genomics prediction tasks, EHR+imaging integration tasks.  
    - *Baselines:*  
      - Single-modality models (omics-only, imaging-only, EHR-only).  
      - CCA-based early fusion methods.  
      - Classical ML fusion (stacking, late fusion, voting).  
      - Deep imaging models without multimodal integration (CNN-only).  
    - *Comparative results from surveyed papers:*  
      - Multimodal models consistently outperform unimodal models on prognosis and treatment-response tasks.  
      - VLMs and multimodal transformers show strongest gains for histopathology + genomics integration.
  **Key Findings and Results**  
  - Clear taxonomy of multimodal integration pipelines for oncology:  
    1. **Preprocessing + harmonization** (handling heterogeneity, missingness, scaling mismatches).  
    2. **Single-modality modelling** (omics representation learning, medical imaging CNNs, EHR NLP).  
    3. **Fusion strategies:**  
       - **Early fusion / data-level** integration for aligned omics data.  
       - **Intermediate fusion / feature-level** using VAEs, contrastive learning, or shared embedding spaces.  
       - **Late fusion / decision-level** where models contribute independently.  
       - **Hybrid deep multimodal transformers** that unify imaging + omics + clinical text.  
    - Multimodal integration improves:  
      - Early cancer detection, especially from pathology + genomics.  
      - Treatment-response prediction (radiomics + genomics + EHR).  
      - Biomarker discovery via omics–imaging representation alignment.  
    - Vision–language models extend the ability to connect clinical notes with pathology or radiology images at scale.  
    - Spatial transcriptomics and single-cell data introduce new spatial–morphological relationships that enhance precision oncology pipelines.
  **Significance**  
  - Provides a unified technical roadmap for multimodal data integration in precision oncology.  
  - Highlights that deep learning–based fusion substantially outperforms single-modality approaches in prognostic and predictive tasks.  
  - Shows how imaging, genomics, and clinical text can be mapped to shared biological representations.  
  - Extends multimodal oncology toward real-world clinical application: diagnosis, treatment response, prognostic modelling, biomarker discovery, and trial recruitment.  
  - Positions VLMs and multimodal transformers as emerging clinical-grade AI systems for oncology.
  **Limitations and Challenges**  
  - Heterogeneity of formats, measurement scales, and coding schemes across modalities.  
  - Missing data, noise, and inconsistency across multi-institution datasets.  
  - Lack of harmonized workflows and standardized multimodal benchmarks.  
  - High computational and storage burdens for imaging + omics fusion.  
  - Limited interpretability of deep multimodal models.  
  - Regulatory constraints, privacy considerations, and clinical deployment barriers.  
  - Need for cross-institution standardization and federated multimodal infrastructures.

- **An Effective Encoding of Human Medical Conditions in Disease Space** (2025)  
  `disease-embedding, multimodal-health-data, comorbidity-analysis, systems-biology, ML, disease-association`
  - This work presents a unified embedding-based framework for representing human diseases in a continuous high-dimensional “disease space” learned from large-scale, sparse, and multimodal health-related data. By embedding diseases as vectors that encode pathological similarity, the method enables quantitative assessment of disease–disease relationships and supports multiple downstream applications including comorbidity discovery, genetic parameter estimation, data-driven disease reclassification, and comorbidity-aware genetic association studies. The authors demonstrate how disease embeddings, inspired by systems biology principles, capture shared etiological mechanisms across conditions and provide a scalable, computationally efficient alternative to traditional phenotypic clustering or co-occurrence–based methods.
  **Biological and Clinical Background**  
  - Human diseases rarely occur in isolation; comorbidity patterns reflect shared genetic, molecular, environmental, or physiological mechanisms.  
  - Large-scale health datasets (EHRs, ICD-coded phenotypes, genetic information) are multimodal, noisy, and sparse.  
  - Traditional co-occurrence statistics fail to capture mechanistic similarity and often ignore complex disease interdependencies.  
  - Disease embeddings map diseases into a latent space where distance encodes biological, genetic, or phenotypic similarity.  
  - This offers a systems biology representation of disease etiology and allows integration of multimodal health signals.
  **Tools, Software, and Databases Used**  
  *(The paper is conceptual but references common methodological infrastructure used in disease embedding research.)*  
  - **Embedding and modelling approaches:**  
    - Word2Vec-like algorithms (Skip-gram, CBOW) adapted for disease sequences in EHRs.  
    - Deep-learning–based embedding models (autoencoders, transformers).  
    - Multimodal integration algorithms linking clinical phenotypes with genetics.  
  - **Clinical and genetic data sources:**  
    - Large EHR repositories (e.g., hospital systems, national registries).  
    - Disease coding systems (ICD9/ICD10).  
    - Genetic association datasets (GWAS summary statistics).  
  - **Computational frameworks referenced:**  
    - Systems biology modelling.  
    - Graph-based disease networks.  
    - High-dimensional embedding learning pipelines.
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - **Evaluation metrics:**  
    - Cosine similarity for disease–disease embedding relationships.  
    - Predictive accuracy for comorbidity inference.  
    - Performance in reconstructing known disease clusters.  
    - Improvement in genetic parameter estimation accuracy.  
  - **Baselines referenced across analyses:**  
    - Co-occurrence frequency models.  
    - Network-based disease similarity measures.  
    - Traditional clustering approaches (hierarchical clustering, PCA).  
    - Classical GWAS without comorbidity integration.  
  - **Evidence of improvement:**  
    - Embedding-based models better capture mechanistic comorbidity patterns.  
    - Improved estimation of disease genetic parameters (heritability, shared variants).  
    - Enhanced classification of diseases into biologically meaningful subgroups.  
    - More accurate comorbidity-aware GWAS adjustments.
  **Key Findings and Results**  
  - Disease embeddings generate a continuous “disease space” that quantitatively encodes pathological similarity.  
  - This representation:  
    - **Identifies hidden disease associations** beyond recorded comorbidities.  
    - **Assists in genetic analysis**, improving estimation of heritability and genetic correlations.  
    - **Enables data-driven disease reclassification**, revealing new clusters reflecting shared biology.  
    - **Transforms GWAS**, allowing comorbidity-aware corrections that reduce confounding.  
  - Provides a unified systems-level view of disease relationships, integrating clinical and genetic evidence.  
  - Supports scalable analyses across millions of patient records.
  **Significance**  
  - Establishes a versatile and powerful framework for deciphering disease relationships using embedding representations.  
  - Bridges clinical observational patterns with underlying biological mechanisms.  
  - Demonstrates potential for building multimodal **disease foundation models** that unify EHRs, genetics, lab tests, and demographics.  
  - Helps move toward holistic modelling of patient health and personalization of diagnosis and risk prediction.  
  - Provides computational tools that could transform pathology, comorbidity analysis, and precision medicine.
  **Limitations and Challenges**  
  - Medical context complexity: disease definitions vary across institutions and evolve over time.  
  - High sparsity and noise in real-world EHR data.  
  - Need for dynamic “online training” to incorporate newly emerging diseases or updated diagnoses.  
  - Validation challenges: interpreting high-dimensional embeddings in clinical settings.  
  - Potential biases from coding practices, demographic heterogeneity, and healthcare access disparities.  
  - Necessity for stronger multimodal fusion (genomics + imaging + clinical data) in future disease foundation models.

- **Kolmogorov–Arnold Networks for Genomic Tasks** (2025)  
  `KANs, genomic-deep-learning, regulatory-genomics, sequence-classification, DNA-generation, benchmarks`
  - This study evaluates Kolmogorov–Arnold Networks (KANs)—a recent deep learning architecture inspired by the Kolmogorov–Arnold representation theorem—as an alternative to multilayer perceptrons (MLPs) in genomic sequence modelling. The authors test Linear KANs (LKANs) and Convolutional KANs (CKANs) within standard genomic pipelines to assess performance in DNA sequence classification and DNA sequence generation. Using three benchmark datasets, they show that LKANs consistently outperform both baseline MLPs and CKANs across most tasks, while achieving strong results with fewer parameters. Ablation experiments reveal that deeper KAN stacks improve accuracy, highlighting the potential of KAN layers as efficient, compact function approximators for regulatory genomics and genome modelling.
  **Biological and Genomic Background**  
  - Genomic sequence modelling underpins tasks such as variant effect prediction, promoter/enhancer recognition, TF-binding modelling, histone mark prediction, splice-site classification, and detection of non-B DNA structures (flipons).  
  - Deep learning in genomics evolved from early CNN/RNN models to transformer-based LLMs (DNABERT, DNABERT-2, Nucleotide Transformer), HyenaDNA (long convolution), and Mamba-based Caduceus.  
  - Despite high performance, transformer models require large computational resources, motivating exploration of smaller architectures capable of competitive accuracy.  
  - KANs use spline-based univariate functions to approximate multivariate functions, potentially offering improved expressivity with fewer parameters.
  **Tools, Models, and Software Used**  
  - **KAN architectures tested:**  
    - *Linear KANs (LKANs)*: KAN layers replacing MLP blocks.  
    - *Convolutional KANs (CKANs)*: KAN-convolution layers replacing CNN blocks.  
  - **Baseline deep learning architectures:**  
    - Standard CNNs, MLP-based classifiers used in genomic sequence tasks.  
  - **Generative modelling tools:**  
    - DDIM (Denoising Diffusion Implicit Model) for DNA generation.  
    - GANs (Generative Adversarial Networks).  
  - **Implementation tools referenced in the field:**  
    - B-spline basis functions, spline activation design, L1-regularized KAN initialization.  
    - Efficient KAN implementations for reducing parameter overhead.  
  **Datasets Used**  
  - **Genomic Benchmarks** — curated regulatory genomics datasets for promoter, enhancer, and TF-binding prediction.  
  - **Genome Understanding Evaluation (GUE)** — multi-task genome modelling evaluation suite used for assessing LLM-style genomic models.  
  - **Flipon Benchmark** — non-B DNA structure datasets (Z-DNA, quadruplex, hairpins) curated by the authors for classification and generative evaluation.  
  **Benchmarks, Evaluation Metrics, and Comparison Baselines**  
  - **Classification metrics:** accuracy, F1-score across benchmark genomic tasks.  
  - **Generative modelling metrics:**  
    - Classification performance of models trained on synthetic sequences (proxy for generative fidelity).  
    - Distribution similarity between real vs generated sequences.  
  - **Baselines:**  
    - CNN-based classifiers.  
    - MLP-based classifiers.  
    - Standard DDIM/GAN generators without KAN layers.  
    - Transformer and LLM performance referenced qualitatively as high-resource baselines.  
  - **Ablation benchmarks:**  
    - Effect of number of KAN layers.  
    - Comparison of LKAN vs CKAN under same parameter budgets.  
  **Key Findings and Results**  
  - **LKANs outperform baseline CNN/MLP architectures** on most GenomicBenchmarks, GUE, and Flipon datasets.  
  - **CKANs achieve competitive performance**, but scaling them to larger parameter counts is difficult due to training overhead from spline functions.  
  - **Deeper KAN stacks improve predictive accuracy**, confirming an architecture–performance correlation.  
  - **Generative modelling:**  
    - KAN-based generative models (within DDIM/GAN pipelines) show strong ability to capture biological sequence patterns, demonstrated by improved classification fidelity on synthetic sequences.  
  - **Parameter efficiency:**  
    - LKAN models achieve strong performance with relatively few parameters, highlighting potential for lightweight genomic models.  
  **Significance**  
  - Introduces KANs as a promising lightweight alternative to MLPs for genomic sequence modelling.  
  - Demonstrates that spline-based function approximators generalize well to regulatory genomics tasks.  
  - Provides a new direction for building compact genomic architectures that avoid transformer-style quadratic scaling.  
  - Suggests potential integration of KAN layers into next-generation genomic foundation models and diffusion-based DNA generators.  
  **Limitations and Future Challenges**  
  - CKANs struggle with parameter scaling and longer training times due to spline overhead.  
  - Current KAN implementations require optimization to maintain stable initialization and variance preservation.  
  - Application to large LLM-scale architectures (e.g., DNABERT-2/Nucleotide Transformer) remains unexplored.  
  - KAN-based models have not yet been benchmarked on ultra-long genomic context tasks (e.g., >10 kb), where transformers or HyenaDNA excel.  
  - Further research is needed to integrate KANs with diffusion models, state space architectures, and multi-species pretraining pipelines.  

- **Artificial Intelligence in Healthcare: A Survival Guide for Internists** (2025)  
  `medical-AI, clinical-decision-support, LLMs, precision-medicine, diagnostics`
  - This review provides an accessible yet rigorous introduction to artificial intelligence (AI) in modern clinical practice, addressing the rapid adoption of machine learning and large language models (LLMs) in diagnostics, prognosis prediction, and personalized medicine. The article distinguishes classical discriminative models from emerging generative approaches and highlights the opportunities and risks at the intersection of AI and internal medicine. With increasing availability of high-volume biomedical data and growing reliance on AI-based tools, the authors emphasize the need for clinicians to understand the functionalities, biases, limitations, and ethical considerations of AI systems—including the well-known issue of hallucinations in LLMs such as ChatGPT.
  **Biological and Clinical Background**  
  - Internal medicine relies heavily on integrating heterogeneous information—clinical exams, imaging, lab tests, EHR narratives—making the domain well-aligned with modern AI methods.  
  - Precision medicine aims to tailor diagnosis and therapy to individual patient profiles, requiring large-scale data integration and predictive modelling.  
  - Clinical AI tools now appear in diagnostic triage, risk scoring, early detection, treatment optimization, and workflow automation.
  **Computational Methods and AI Approaches**  
  - **Discriminative models:** logistic regression, random forests, gradient boosting, classical neural networks for classification and regression tasks.  
  - **Generative models:**  
    - Large language models (GPT-based, LLaMA-based) for clinical text synthesis, summarization, and question answering.  
    - Multimodal AI models integrating imaging + text + structured EHR.  
  - **Key distinctions:**  
    - Discriminative AI: predicts target labels from clinical features.  
    - Generative AI: produces free-text outputs, explanations, and synthetic data, but subject to hallucination risks.  
  - **Critical methodological considerations:** overfitting, data leakage, non-representative datasets, label noise, model calibration, and domain shift.
  **Tools, Software, and Data Sources Used in Medical AI**  
  - **LLMs mentioned**: ChatGPT, other recent large language models used in clinical contexts.  
  - **Common clinical AI pipelines** rely on:  
    - Electronic Health Records (EHR) systems.  
    - Medical imaging (radiology, histopathology).  
    - High-throughput biological data (genomics, proteomics, lab trends).  
    - Time-series biosignals (ECG, vital signs).  
  - **Programming ecosystems** typically include Python frameworks (TensorFlow, PyTorch, Scikit-learn), but the paper focuses conceptually rather than providing specific toolboxes.
  **Datasets Referenced or Implicitly Used in Clinical AI**  
  *(Note: The review is conceptual and does not evaluate AI on specific datasets, but it references the typical types used in medical AI research)*  
  - Clinical EHR datasets (ICU databases like MIMIC; hospital information systems).  
  - Radiology datasets (MRI, CT, X-ray).  
  - Omics datasets for precision medicine (genomic panels, transcriptomics).  
  - Real-world data from hospital clinical workflows.
  **Benchmarks, Evaluation Metrics, and Baselines**  
  *(The article highlights evaluation principles rather than specific experiments)*  
  - **Metrics commonly required in medical AI:**  
    - AUROC, AUPRC for diagnosis/risk stratification.  
    - Sensitivity/specificity depending on clinical use-case.  
    - Calibration curves to assess clinical reliability.  
  - **Baselines:** traditional clinical scores, classical ML models, and non-AI clinical guidelines.  
  - **Model validation guidelines:**  
    - Clear train/validation/test splits.  
    - Avoiding data leakage from EHR temporality.  
    - External validation across hospitals.
  **Key Insights and Results**  
  - LLMs drastically reduce entry barriers for clinicians but introduce high hallucination risk when used for clinical reasoning without verification.  
  - Discriminative AI remains more reliable for structured prediction tasks; generative AI offers flexibility but needs guardrails.  
  - Data quality and annotation remain the main limitations in developing trustworthy medical AI.  
  - AI can accelerate precision medicine through better prediction of disease trajectories and treatment responses.  
  - Adoption in internal medicine requires clinical understanding of algorithmic strengths and failure modes.
  **Significance**  
  - Provides a conceptual “survival guide” enabling internists to critically assess and safely use AI tools.  
  - Bridges the knowledge gap between AI research and clinical decision-making.  
  - Encourages integration of AI literacy into routine medical training.  
  - Highlights the importance of combining human clinical judgment with algorithmic assistance rather than replacing it.
  **Limitations and Challenges**  
  - LLM hallucinations present major risks in medical decision support.  
  - Lack of transparency in proprietary models limits explainability and trust.  
  - Scarcity of high-quality labelled clinical datasets hinders robust model development.  
  - Ethical and legal challenges include accountability, bias, data privacy, consent, and regulatory oversight.  
  - Needs standardized clinical AI evaluation frameworks and external validation before deployment.  

- **Pre-Meta: Priors-Augmented Retrieval for LLM-Based Metadata Generation** (2025)  
  `genomics-metadata, RAG, LLMs, ontology-integration, data-harmonization`
  - This paper introduces **Pre-Meta**, a model-agnostic retrieval-augmented generation (RAG) pipeline designed to automate and enhance metadata annotation for genomic and biomedical datasets. The method tackles a major bottleneck in modern genomics: while sequencing and high-throughput technologies generate massive volumes of data, the corresponding metadata remain fragmented, heterogeneous across repositories, and often incomplete. Pre-Meta improves metadata quality by incorporating *priors*—pre-existing metadata labels, ontology terms, and controlled vocabularies—directly into the retrieval mechanism, enabling large language models to produce more accurate and ontology-aligned annotations without requiring finetuning.
  **Biological and Genomic Background**  
  - Biomedical repositories such as GEO, ArrayExpress, and cBioPortal collectively host millions of datasets (gene expression, sequencing assays, cancer genomics).  
  - These datasets depend critically on metadata fields describing sample type, experimental design, organism, study type, disease classification, etc.  
  - Metadata heterogeneity stems from incompatible formats (e.g., **MAGE-TAB** vs. **MINiML**) and non-harmonized ontologies, complicating dataset discovery and secondary analyses.  
  - Manual annotation is labour-intensive, error-prone, and inconsistent—creating a barrier to FAIR genomics (Findable, Accessible, Interoperable, Reusable).
  **Computational Approach and Methodology**  
  - **Pre-Meta** builds on the RAG paradigm but enriches retrieval using *priors*:  
    - pre-generated metadata tags,  
    - structured ontology concepts (EFO, OBI, SNOMED),  
    - related contextual fields from existing repositories.  
  - The pipeline is **LLM-agnostic** and **schema-agnostic**, meaning it can operate with any large language model and adapt to any metadata schema (ArrayExpress, GEO, Beacon v2).  
  - Workflow:  
    1. Extract text segments from publications and datasets (Europe PMC, ArrayExpress).  
    2. Retrieve ontology-aligned contextual documents.  
    3. Augment the prompt with priors.  
    4. Generate metadata using an LLM.  
    5. Map output to controlled vocabulary terms.  
  - Designed to avoid LLM hallucinations by constraining outputs using controlled vocabularies and curated priors.
  **Tools, Software, and Databases Used**  
  - **Repositories:**  
    - ArrayExpress (MAGE-TAB format),  
    - GEO (MINiML format),  
    - cBioPortal,  
    - Research Data Repository Registry (re3data),  
    - Europe PubMed Central (Europe PMC).  
  - **Ontologies & Standards:**  
    - Experimental Factor Ontology (EFO),  
    - Ontology for Biomedical Investigations (OBI),  
    - SNOMED CT, MeSH,  
    - GA4GH Beacon v2 semantic framework.  
  - **Models Tested:**  
    - GPT-4o mini,  
    - LLaMA 8B,  
    - Mistral 7B.  
  - **Pipeline code:** Available at GitHub (LLMDap).
  **Datasets Used in Evaluation**  
  - 1,500 scientific papers from Europe PMC corresponding to real-world ArrayExpress studies.  
  - Five representative metadata fields sampled from these datasets (e.g., study type, organism, tissue/cell type, technology platform).
  **Benchmarks, Evaluation Metrics, and Baselines**  
  - **Baselines:**  
    - Standard RAG (vanilla retrieval + generation).  
    - Direct LLM inference without retrieval.  
  - **Evaluation Metrics:**  
    - Annotation accuracy (exact-match with ground-truth labels).  
    - Controlled vocabulary alignment (ontology-consistent output).  
  - **Performance Gains:**  
    - Pre-Meta outperformed standard RAG by:  
      - **+23% (GPT-4o mini)**  
      - **+72% (LLaMA 8B)**  
      - **+75% (Mistral 7B)**  
    - Achieved this **without** finetuning or prompt engineering.
  **Key Results and Insights**  
  - Using ontology-aligned priors significantly improves metadata generation quality across all LLMs.  
  - The benefits are *model-independent*, demonstrating that priors constrain LLMs effectively even for smaller open-weight models.  
  - Pre-Meta reduces hallucinations by grounding predictions in curated vocabulary.  
  - The system supports harmonization across heterogeneous repositories, reducing manual curation workload.
  **Significance**  
  - Offers a generalizable, low-cost approach to improve genomic metadata curation at scale.  
  - Bridges multiple repositories with incompatible standards, enabling interoperable FAIR data.  
  - A strong candidate for automating metadata extraction across global genomics infrastructures (including federated Beacon networks).  
  - Supports downstream tasks such as dataset discovery, integrative analyses, and ontology-based indexing.
  **Limitations and Challenges**  
  - Strongly depends on the completeness and correctness of ontology priors.  
  - Metadata fields with sparse or ambiguous descriptions remain challenging.  
  - Retrieval errors or irrelevant context may degrade performance.  
  - Some metadata concepts do not map cleanly across repositories (schema inconsistencies remain).  
  - Global harmonization still requires consensus across the genomics community.

- **BV-BRC: A Unified Bacterial and Viral Bioinformatics Resource with Expanded Functionality and AI Integration** (2025)  
  `pathogen-genomics, bacterial-viruses, bioinformatics-platforms, multimodal-data, RAG, AI-copilots`
  - BV-BRC is a large-scale integrative bioinformatics platform consolidating bacterial, archaeal, viral, bacteriophage, and microbiome genomics into a unified, FAIR-compliant ecosystem. Hosting more than **14 million publicly available genomes**, BV-BRC provides standardized annotations, curated metadata, and cross-linked multi-omics resources including transcriptomics, protein interactions, 3D structures, antigenic epitopes, and epidemiological metadata. The latest update introduces expanded analysis services, state-of-the-art genomics pipelines, and an AI-powered assistant, **BV-BRC Copilot**, which leverages LLMs with retrieval-augmented generation to support natural-language queries, workflow guidance, and knowledge integration for pathogen research.
  **Biological Background**  
  - Pathogen genomics (bacteria, viruses, phages) underpins surveillance, outbreak tracking, antimicrobial resistance studies, and comparative genomics.  
  - BV-BRC consolidates heterogeneous genomic, transcriptomic, functional annotation, and epidemiological datasets into a single harmonized infrastructure.  
  - Supports both basic research (phylogenomics, functional genomics, multistrain comparison) and translational research (pathogen typing, vaccine/antigen exploration, outbreak monitoring).
  **Computational Methods and System Architecture**  
  - Unified data architecture linking genomes → genes → annotations → metadata → multi-omics records.  
  - Incorporates AI-driven workflow assistance through **BV-BRC Copilot**, a natural-language interface using:  
    - Large language models (LLMs)  
    - Retrieval-Augmented Generation (RAG)  
    - Knowledge-grounded query resolution  
  - Expanded computational services include comparative genomics, metagenomics, taxonomic classification, RNA-seq processing, and molecular docking.  
  - Scalable backend supporting >20,000 analysis jobs/month with >95% success rate.
  **Tools, Software, Pipelines, and Services**  
  BV-BRC integrates or updates 33+ analysis workflows including:  
  - **Genome assembly & annotation:** updated pipelines using state-of-the-art assemblers and gene callers.  
  - **Rapid comparative genomics:** SNP-based comparison, pan-genome analysis, phylogenomic reconstruction.  
  - **Metagenomics services:** read mapping, taxonomic classification, wastewater pathogen analysis.  
  - **RNA-seq analysis:** differential expression, functional annotation.  
  - **Viral genomics:** viral subspecies classification, assembly, annotation.  
  - **Molecular docking:** ligand–protein interaction modeling for therapeutic exploration.  
  - **Interactive tools:** genome browsers, phylogenetic viewers, protein structure visualization, heatmaps, synteny plots.  
  - **Programmatic access:** REST APIs, CLI, private workspaces.
  **Datasets Integrated into BV-BRC**  
  - **14+ million bacterial & viral genomes**, updated monthly.  
  - Curated **multi-omics datasets**:  
    - Transcriptomics (RNA-seq)  
    - Protein–protein interactions  
    - Protein structures  
    - Epitope collections  
  - FAIR-linked metadata from original publications and source repositories.  
  - Outbreak-tracking datasets with spatiotemporal epidemiological information.  
  - User-uploaded private datasets with secure analysis and sharing.
  **Benchmarks, Evaluation Metrics, and Performance Monitoring**  
  While BV-BRC is not a predictive model benchmark, the paper reports operational performance metrics as validation:  
  - **>20,000 analysis jobs/month** with **>95% completion success rate**.  
  - **>70,000 registered active users** spanning researchers, clinicians, epidemiologists, and data scientists.  
  - **~750 citations in the past year**, indicating global adoption in infectious disease genomics.  
  - Monthly integration of **thousands of new genomes**, validated via pipeline consistency checks.  
  - AI Copilot evaluated for correctness through internal curation and tool-guided RAG safeguards (no direct genome exposure to LLM).
  **Key Results and New Capabilities**  
  - Substantial expansion of viral genomics and epidemiology services (e.g., wastewater pathogen tracking).  
  - Addition of molecular docking for therapeutic hypothesis generation.  
  - Significant pipeline upgrades for assembly, annotation, and taxonomic classification.  
  - Introduction of an intelligent LLM-driven Copilot that improves accessibility for non-experts and speeds up exploratory analyses.  
  - Stronger integration of multi-omics and metadata enabling more complex comparative and functional analyses.
  **Significance**  
  - Provides the most comprehensive unified platform for bacterial and viral genomics worldwide.  
  - Bridges research and public health by enabling rapid outbreak analysis, pathogen surveillance, and comparative genomic insights.  
  - AI integration lowers the barrier to advanced bioinformatics, supporting both beginners and experts.  
  - Advances FAIR principles through harmonized metadata and open programmatic access.
  **Limitations and Challenges**  
  - Integration across heterogeneous data sources may cause metadata inconsistencies despite curation.  
  - AI Copilot depends on RAG correctness; retrieval noise may degrade response quality.  
  - Resource demands scale with growing genomic datasets (storage, compute, curation effort).  
  - Some pathogen classes or rare viral species may still be underrepresented.  
  - Requires ongoing maintenance of tools as sequencing technologies evolve.

- **PaxDb v6.0: Reprocessed, LLM-selected, Curated Protein Abundance Data Across Organisms** (2025)  
  `proteomics, protein-abundance, mass-spectrometry, data-curation, LLM-classification, FragPipe, cross-species-comparison`
  - PaxDb v6.0 represents a major expansion of the global reference compendium for healthy-state protein abundance across organisms. The new release nearly doubles its coverage to **1639 datasets across 392 species**, integrating diverse mass-spectrometry experiments into harmonized organism- and tissue-level abundance profiles. Version 6.0 introduces the first large-scale automated reprocessing of public proteomics raw files using a standardized MS pipeline, combined with LLM-based project pre-selection to improve curation throughput. The resource offers organism-wide and tissue-resolved abundance maps expressed in ppm units, enabling consistent cross-study and cross-species comparisons under controlled healthy conditions.
  **Biological Background**  
  - Proteomics provides direct quantification of the cellular functional machinery—not simply gene potential or mRNA levels.  
  - Protein abundance is decoupled from transcript levels and spans a dynamic range of 10⁶–10¹⁰, making it a crucial layer for understanding baseline physiology, biomarker discovery, and perturbation responses.  
  - High-quality healthy-state proteome baselines underpin studies of differential expression, PPIs, immunopeptidomics, and tissue-specific functional characterization.  
  - Public MS data are abundant but heterogeneous in metadata, processing pipelines, and technical quality, limiting reuse. PaxDb standardizes and integrates these datasets.
  **Computational Methods & System Architecture**  
  - **End-to-end MS reprocessing pipeline** built on the FragPipe framework:  
    - Raw file ingestion and grouping  
    - Automated metadata parsing and linking  
    - Consistent peptide/protein identification using up-to-date reference genomes  
    - DIA/DDA support via DIA-NN and FragPipe modules  
    - STRING-based PPI-informed quality scoring  
  - **LLM-ensemble classifiers** used to semi-automate ProteomeXchange project selection:  
    - Trained to distinguish valid whole-proteome healthy datasets from irrelevant or out-of-scope studies  
    - Reduces manual curation effort and increases detection of suitable datasets  
  - **Ontology-driven metadata harmonization** (UBERON, PO, CLO, CL, BTO).  
  - Orthology mapping via **eggNOG** enabling cross-organism comparisons.  
  - Web interface for end-user submission: peptide-level abundance calculation, dataset QC, and comparison to PaxDb reference profiles.
  **Tools, Software, and Processing Frameworks Used**  
  - **FragPipe** (core MS processing engine)  
  - **DIA-NN** (DIA quantification)  
  - **STRING** (PPI network scoring)  
  - **LLM ensemble models** for dataset triage  
  - **Ontology systems:** UBERON, Plant Ontology, CLO, CL, BTO  
  - **Orthology resource:** eggNOG  
  - **PX accession integration:** ProteomeXchange IDs for provenance  
  - **PaxDb web platform**: interactive exploration, QC, peptide-level tool
  **Datasets Integrated into v6.0**  
  - **1639 public proteomics datasets** across **392 species**, including all kingdoms of life.  
  - Tissue-level and organism-wide proteomes under strictly defined healthy, wild-type conditions.  
  - Includes data sourced from:  
    - ProteomeXchange repositories  
    - PeptideAtlas  
    - ProteomicsDB  
    - Large single-organism consortia  
  - Full provenance: PubMed IDs, PX project links, metadata, orthology group assignment.  
  - Excluded categories (by design): cross-linking experiments, fractionation, disease conditions, environmental/multispecies samples, PTM-focused datasets, biochemical fraction studies, synthetic/ancient samples.
  **Benchmarks, Evaluation Metrics, and Performance Validation**  
  - **Quality scoring** based on STRING PPI coherence across protein abundance profiles.  
  - Technical reproducibility monitored across reprocessed datasets using standardized FragPipe workflows.  
  - Accuracy and robustness of LLM classifiers validated on manually curated benchmark subsets of PX projects.  
  - Coverage and completeness metrics:  
    - Dataset retention rate after QC  
    - Fraction of proteome quantified per species  
    - Orthology completeness across kingdoms  
  - User-facing QC metrics for uploaded datasets:  
    - Peptide-level recovery  
    - Abundance distribution shape  
    - Cross-reference to PaxDb organism baseline
  **Key Results & Improvements in v6.0**  
  - Expanded to **392 species** (nearly ×2 since v5.0).  
  - **First large-scale unbiased reprocessing** of MS raw files across public proteomics repositories.  
  - Robust, ontology-driven metadata standardization enabling FAIR access.  
  - Semi-automated dataset curation using LLM classifiers greatly increases coverage and reduces manual screening workload.  
  - Integrated cross-species comparison enabled by updated orthology mappings.  
  - New peptide-level tool supports users in benchmarking their own data against PaxDb references.
  **Significance**  
  - Provides the largest healthy-state cross-organism protein-abundance reference available.  
  - Standardized reprocessing eliminates inconsistencies caused by heterogeneous original pipelines.  
  - Enables reliable cross-study integration and comparative proteomics across evolutionary scales.  
  - Supports downstream biomedical applications: biomarker discovery, physiological baseline modeling, PPI interpretation, systems biology.  
  - Demonstrates practical, high-impact use of LLMs for biomedical data curation and repository scaling.
  **Limitations & Challenges**  
  - Despite LLM triage, metadata incompleteness in PX submissions still limits automated selection.  
  - Healthy-state datasets dominate; disease and perturbation conditions are intentionally excluded, limiting direct clinical use.  
  - Ontology harmonization remains complex across hundreds of species and experimental designs.  
  - Reprocessing is computationally demanding and must be continuously updated as MS technologies evolve.  
  - Some organisms remain underrepresented due to limited high-quality data availability.


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

- *spaLLM: Enhancing Spatial Domain Analysis in Multi-Omics Data through LLM Integration*, 2024  
  `llms, scgpt, spatial-transcriptomics, multi-omics, gnn, spatial-domain-analysis`  
  — Introduces spaLLM, the first spatial multi-omics domain analysis method integrating single-cell foundation models (scGPT) with GNNs and multi-view attention. The model leverages LLM-derived gene embeddings to overcome sparse spatial gene expression and improves spatial domain resolution across RNA, chromatin, and protein modalities. Benchmarked on four datasets, spaLLM surpasses eight SOTA methods in supervised metrics, demonstrating the power of LLM-enhanced spatial biology.


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

- *"Multimodal large language models in medical research and clinical practice: Development, applications, challenges and future"*, 2024  
  `multimodal-llms, medical-ai, clinical-informatics, trustworthy-ai, healthcare-systems`  
  — A comprehensive review of multimodal LLMs in healthcare, covering vision–text–signal–EHR fusion, clinical diagnostic applications, infrastructure requirements, and ethical/compliance challenges. Provides a system-level framework for integrating MLLMs into intelligent healthcare workflows, highlighting bottlenecks such as data silos, fusion strategies, compute limitations, and regulatory constraints.


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

- *"Generating pregnant patient biological profiles by deconvoluting clinical records with electronic health record foundation models"*, 2025  
  `ehr-foundation-models, clinical-ai, proteomics-generation, fm-representations, biomedical-reliability`  
  — Uses state-of-the-art EHR foundation models to reconstruct proteomic profiles from clinical data, generating 206 protein expression levels without laboratory assays. Highlights biological pathway enrichment, FM-derived patient state reconstruction, and a proteomic signature for gestational diabetes. Demonstrates how clinical FMs can reliably infer biological states, offering a cost-efficient alternative to traditional omics profiling.

- *Exploring the Impact of Large Language Models on Disease Diagnosis*, Almubark (2024)  
  `llms, clinical-diagnosis, medical-nlp, gpt4, llama, clinical-evaluation`  
  — A systematic review evaluating how LLMs such as GPT-4, GPT-3.5, LLaMA, and Bard assist in disease diagnosis across chronic, respiratory, oncological, and rare conditions. The study assesses multimodal data sources (clinical text, medical images, genomic data) and reports improvements in diagnostic accuracy and decision support effectiveness.

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
