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

- **"The Large Language Models on Biomedical Data Analysis: A Survey"**, 2025, JBHI  
  `llm, genomics, proteomics, transcriptomics, radiomics, single-cell, drug-discovery`  
  — This comprehensive survey reviews the rapid expansion of LLMs across biomedical data domains. It summarizes LLM foundations (transformer architectures, tokenization strategies, pretraining corpora), biomedical datasets (omics, imaging, EHR), and key frameworks for model adaptation (prompt engineering, fine-tuning, domain-specific training). The survey categorizes LLM applications in genomics (variant interpretation, regulatory-element prediction), proteomics (function annotation), transcriptomics (gene-expression modeling), radiomics (image–text translation), and drug discovery (target mining, molecular representation). It further reviews evaluation metrics (perplexity, AUROC, Recall@K, F1) and highlights major challenges such as domain shift, hallucinations, limited benchmarks, clinical safety, and the lack of multimodal grounding. The survey positions LLMs as promising yet immature tools requiring rigorous biomedical alignment.

- **"Foundation Models for Bioinformatics"**, 2023  
  `foundation-models, transformers, genomics, proteomics`  
  — A perspective review on transformer-based foundation models for biological sequence and structure analysis. The paper compares general-purpose LLMs (GPT, T5) with domain-adapted models such as DNABERT, Geneformer, ESM, ProtGPT2, highlighting differences in tokenization (k-mers, amino acids), pretraining objectives (masked modeling, next-token prediction), and data sources (UniProt, ENCODE, RefSeq). It describes transfer learning strategies, prompt-based adaptation, and techniques for reducing hallucinations in biomedical reasoning. Applications span motif detection, variant pathogenicity estimation, protein stability prediction, and multi-omics integration. The review underscores the need for biologically grounded evaluation frameworks and improved interpretability tools.

- **"Revolutionizing Personalized Medicine with Generative AI: A Systematic Review"**, 2024  
  `generative-ai, dgms, llms, precision-medicine, synthetic-data`  
  — A systematic review of generative models in precision medicine covering GANs, variational autoencoders, diffusion models, and LLMs. It analyzes their use in synthetic EHR/omics data generation, early disease detection, drug-response modeling, and individualized treatment-effect prediction. The paper highlights how synthetic data can mitigate privacy concerns and support rare-disease modeling. It evaluates performance via fidelity metrics (Fréchet distance, NMSE), downstream predictive accuracy, and clinical calibration. Limitations include instability of GAN training, mode collapse, biases in generative distributions, and uncertainties in LLM-driven diagnoses. The review proposes multimodal generative FMs for integrating genomics, imaging, and longitudinal clinical data.

- **"Large Language Models With Applications in Bioinformatics and Biomedicine"**, 2025, IEEE JBHI  
  `llms, foundation-models, multimodal-biomedical-ai, molecular-modeling`  
  — A guest editorial summarizing state-of-the-art LLM-based developments in molecular biology and medicine. It highlights advances in molecular property prediction (LLM-driven molecular embeddings), drug–herb interaction modeling, protein/RNA functional prediction using transformer encoders, multimodal fusion (text + sequence + structure), and clinical AI. Emerging solutions include contrastive learning for multimodal alignment, knowledge distillation for efficiency, and attention/saliency maps for interpretability. The editorial underscores persistent challenges in data scarcity, multimodal harmonization, and transparent biological reasoning.

- **"Progress and Opportunities of Foundation Models in Bioinformatics"**, 2024  
  `foundation-models, llms, genomics, proteomics, multimodal-biology`  
  — A comprehensive survey outlining the evolution of biological foundation models across DNA, RNA, protein, and multimodal datasets. It reviews architectures such as DNABERT, Geneformer, ESM, ProtT5, and multi-omics transformers trained on heterogeneous biological corpora. Applications include sequence labeling, protein structure prediction, variant impact modeling, and gene regulatory network inference. The survey details pretraining techniques (masked-token modeling, contrastive learning, multitask learning) and evaluates performance on standard benchmarks. Challenges include noisy biological data, lack of standardized evaluation pipelines, limited interpretability, and domain bias. Future directions include sparse attention for scalability, cross-species generalization, and explainable biological embeddings.

- **"Challenges in AI-Driven Biomedical Multimodal Data Fusion and Analysis"**, 2024  
  `multimodal-learning, llms, biomedical-fusion, interpretability, meta-learning`  
  — This survey examines the rapidly growing field of multimodal biomedical AI, focusing on fusion of molecular, cellular, imaging, and EHR-based modalities. It categorizes fusion strategies into early fusion (feature concatenation), late fusion (ensemble-based), and deep fusion (cross-attention, joint embedding learning). The paper analyzes transformer-based multimodal architectures integrating omics and imaging, and highlights how LLMs can be used for knowledge-guided integration and metadata-aware representation learning. Key challenges include dataset imbalance, missing modalities, sample heterogeneity, privacy-preserving fusion, and the difficulty of interpreting cross-modal attention maps. The review proposes meta-learning and knowledge-graph–enhanced fusion for future scalable multimodal systems.

- **"Biomedical Natural Language Processing in the Era of Large Language Models"**, 2025, Annual Review of Biomedical Data Science  
  `biomedical-nlp, llms, ehr, generative-ai, clinical-text`  
  — A high-level review covering advances in biomedical NLP driven by large language models. It analyzes domain-specific LLMs (BioGPT, ClinicalBERT, Med-PaLM, GatorTron) and frontier general LLMs, focusing on tasks such as clinical summarization, diagnosis-support reasoning, medical NER, relation extraction, temporal extraction, and population-level health analytics. The survey evaluates hallucination risks, factuality issues, omission errors, privacy constraints, and challenges in aligning LLM outputs with clinical documentation standards. It highlights multimodal integration (radiology + EHR + genomics) and discusses pathways toward clinically safe, reliable, and interpretable biomedical LLMs.

- **"Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions"**, 2024  
  `foundation-models, healthcare-fm, llms, multimodal-ai, clinical-ai`  
  — A comprehensive healthcare-focused survey analyzing clinical foundation models (HFMs) across text, imaging, and structured EHR data. It evaluates architectures such as encoder–decoder FMs, multimodal transformers, and retrieval-augmented clinical LLMs. Applications include patient-triage prediction, report generation, diagnosis support, and disease risk modeling. Challenges addressed include data quality, noisy/biased clinical notes, compute cost, fairness/robustness issues, and real-time deployment in hospitals. The study outlines future opportunities including federated clinical FMs, reinforcement learning with clinician feedback (RLHF), and multimodal diagnostic agents.

- **"Bridging Artificial Intelligence and Biological Sciences: A Comprehensive Review of Large Language Models in Bioinformatics"**, 2024  
  `llms, bioinformatics, survey, protein-structure, genomics, drug-discovery`  
  — This review discusses LLM applications across protein structure prediction, RNA/DNA modeling, variant interpretation, biomedical literature mining, and AI-driven drug design. It highlights progress from traditional statistical sequence models to transformer-based biological LLMs and examines how LLMs capture structural constraints, evolutionary information, and biochemical patterns. The review assesses domain adaptation techniques (continual pretraining, instruction tuning), explains shortcomings such as hallucination, domain bias, and lack of mechanistic interpretability, and emphasizes future directions such as hybrid symbolic–neural reasoning and cross-modal biological knowledge integration.

- **"A Survey for Large Language Models in Biomedicine"**, 2024  
  `biomedicine, llms, multimodal-llms, zero-shot-learning, fine-tuning`  
  — A large-scale survey synthesizing findings from 484 biomedical AI/LLM publications. It categorizes LLM contributions into diagnostic reasoning, generative drug design, clinical decision support, biomedical NER/RE, causal knowledge mining, and personalized medicine. The study pays special attention to zero-shot and few-shot evaluation, reporting that while frontier LLMs excel at general reasoning, they underperform in fine-grained biomedical tasks requiring grounded domain knowledge. Adaptation techniques such as LoRA, adapters, prompt tuning, and multimodal alignment are reviewed. Key challenges include privacy, interpretability, faulty assumptions in training corpora, and the lack of biomedical safety benchmarks. The paper proposes federated and privacy-preserving biomedical LLM ecosystems.

- **"Multimodal Large Language Models in Health Care: Applications, Challenges, and Future Outlook"**, 2025  
  `multimodal-llms, clinical-ai, radiology, pathology, genomics, sensor-data`  
  — A comprehensive survey covering multimodal healthcare LLMs that jointly process text, imaging, signals, and omics. It reviews architectures such as text–vision transformers, radiology report generation models, pathology–omics fusion frameworks, and ICU multimodal monitoring agents. The paper evaluates alignment techniques (cross-attention, contrastive embedding, joint latent spaces) and identifies key clinical applications such as differential diagnosis, report–image consistency checking, surgical planning, and multimodal risk prediction. Challenges include data silos, high compute demands, hallucination amplification across modalities, regulatory barriers, and the absence of standardized clinical benchmarks. Future directions highlight unified hospital-scale AI agents, safety-guided multimodal alignment, and clinically interpretable embeddings.

- **"The Development Landscape of Large Language Models for Biomedical Applications"**, 2025  
  `biomedical-llms, model-development, transformer-architectures, clinical-nlp, survey`  
  — A PRISMA-guided review analyzing 82 biomedical-specific LLMs released since 2022. It maps evolution trends in model architecture (dominance of decoder-only Llama-like models), biomedical dataset construction (PubMed, PMC, EHR), tokenizer designs (subword vs. biomedical-aware tokenization), and parameter-efficient fine-tuning strategies (LoRA, adapters, prefix-tuning). The survey compares performance across tasks such as NER, QA, summarization, concept linking, and medical reasoning. Highlighted challenges include privacy-restricted corpora, reproducibility issues from undocumented training details, model bias, and inadequacies in factuality evaluation. The review calls for transparent documentation standards and clinically aligned safety benchmarks.

- **"Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics"**, Nature Methods, 2024/2025  
  `genomic-foundation-models, dna-llm, masked-language-modeling, variant-prediction`  
  — Introduces the Nucleotide Transformer (NT), a suite of DNA foundation models ranging from 50M to 2.5B parameters. NT is pretrained on the human reference genome, 3,202 human genomes, and 850 multi-species genomes using masked nucleotide modeling. The models generate context-aware nucleotide embeddings and are evaluated across 18 genomics benchmarks under 10-fold cross-validation, outperforming DNABERT, Enformer-lite, and supervised baselines. Analyses include scaling laws, attention distribution over regulatory elements, zero-shot variant prioritization, and cross-species generalization. Results show that larger NT models infer subtle regulatory patterns and achieve strong performance in promoter, enhancer, and splice-site prediction tasks.

- **"MutBERT: Probabilistic Genome Representation Improves Genomics Foundation Models"**, 2025  
  `genomics-foundation-models, snp-representation, masked-language-modeling, population-genomics`  
  — MutBERT replaces deterministic genome representation with probabilistic allele-frequency–based encoding to leverage population-scale variation. This reduces redundancy in invariant genomic regions and focuses model capacity on polymorphic areas rich in regulatory signals. MutBERT is pretrained with masked-language modeling and evaluated on variant effect prediction, enhancer annotation, chromatin-state inference, and regulatory element classification. Compared with DNABERT-2, Nucleotide Transformer, HyenaDNA, and MambaDNA, MutBERT shows superior SNP-aware modeling, improved separation of pathogenic vs. benign mutations, and stronger cross-population generalization on biobank datasets.

- **"Benchmarking DNA Large Language Models on Quadruplexes"**, 2024  
  `dna-llms, dnabert2, hyenadna, mamba-dna, g-quadruplex`  
  — A systematic benchmark evaluating transformer-based (DNABERT-2), long-convolution (HyenaDNA), and state-space (MambaDNA, Caduceus) foundation models in predicting G-quadruplexes (non-B DNA secondary structures). The benchmark uses whole-genome windows and curated quadruplex datasets, assessing performance via F1, MCC, and AUROC. Results show DNABERT-2 and HyenaDNA achieve highest F1/MCC, while HyenaDNA captures distal enhancer-associated and intronic quadruplexes missed by transformer-only models. The study reveals complementary strengths of different long-context architectures and highlights model-selection considerations for noncanonical DNA-structure prediction.

- **"Distinguishing Word Identity and Sequence Context in DNA Language Models"**, 2024  
  `dna-llms, dnabert, tokenization, sequence-context, k-mers`  
  — This analytical study investigates how DNABERT’s overlapping k-mer tokenization affects learning. The authors propose a benchmark that separates identity modeling (memorization of k-mers) from context modeling (long-range dependency capture). Using non-overlapping next-token prediction and embedding-space analyses, they show DNABERT overemphasizes k-mer identity, leading to redundancy and weakened contextual understanding. Visualization of embedding manifolds reveals strong clustering around repeated k-mers. The study motivates alternative tokenization strategies—including sparse k-mers, BPE-style genomic tokens, and hierarchical encodings—to improve biological relevance and reduce token redundancy.

- **"Large Language Models in Genomics: A Perspective on Personalized Medicine"**, 2024  
  `llms, genomics, precision-medicine, diagnostic-support`  
  — A perspective article analyzing how LLMs reshape clinical genomics workflows, particularly variant interpretation and genotype–phenotype association extraction. LLMs assist clinicians by generating evidence-based variant summaries, extracting gene–disease relationships from literature, and integrating EHR notes with genetic test results. The paper covers risk factors such as hallucinated pathogenicity claims, ancestry-driven biases, and overconfident reasoning. It contrasts LLM-based reasoning with ACMG/AMP standardized variant-classification guidelines and recommends hybrid systems integrating knowledge graphs, curated variant databases, and clinician oversight.

- **"Deep Learning for Genomics: From Early Neural Nets to Modern Large Language Models"**, 2025  
  `genomics-review, genomic-deeplearning, dna-llms, multimodal-genomics`  
  — A historical review charting the transition from early CNN/RNN genomic models to modern transformer-based genomic foundation models (DNABERT, NT, Enformer, HyenaDNA). The article emphasizes how attention, long-context modules, and multi-species pretraining enable models to capture distal regulatory interactions, 3D chromatin structure, and enhancer–promoter communication. Benchmarks such as DeepSEA, ENCODE, Basenji, and CAGI competitions are reviewed. Limitations include sparse labels, enormous sequence lengths, limited experimental validation, and interpretability challenges. Future directions include multimodal DNA–epigenome–transcriptome FMs and mechanistic hybrid modeling.

- **"NextVir: Enabling Classification of Tumor-Causing Viruses with Genomic Foundation Models"**, 2024  
  `viral-classification, genomic-foundation-models, dnabert, hyenadna, nucleotide-transformer`  
  — NextVir leverages genomic foundation models (DNABERT-S, HyenaDNA, NT) to classify viral reads by species and oncogenicity. By training on curated viral genomes and nonviral backgrounds, the system achieves strong generalization to unseen viral strains and noisy reads. Benchmarked against CNN/RNN viral classifiers, NextVir outperforms them across accuracy, F1, and MCC metrics. The method shows robustness on short read lengths (<150 bp), making it suitable for metagenomic surveillance and cancer-associated viral discovery. The study highlights how DNA FMs detect subtle viral oncogenic patterns absent from classical models.

- **"ViraLM: Empowering Virus Discovery Through the Genome Foundation Model"**, 2025  
  `viral-detection, dnabert2, metagenomics, short-contigs, genome-foundation-models`  
  — ViraLM builds on DNABERT-2 to detect viral fragments in metagenomic assemblies. Pretrained on ~50k viral genomes plus nonviral background, the model learns discriminative sequence patterns enabling accurate viral identification even in short contigs (<1 kb). The benchmark spans IMG/VR, RefSeq, and real metagenomic datasets, where ViraLM surpasses protein-based tools (geNomad, VIBRANT) and nucleotide baselines (VirRep, DeepVirFinder), achieving up to +22% F1 improvement. The system identifies novel viral clades missed by existing pipelines and highlights the advantage of FM-driven sequence embeddings for viral ecology and pathogen surveillance.

- **"The Role of Chromatin State in Intron Retention: Leveraging Large-Scale Deep Learning Models"**, 2024, PLOS Computational Biology  
  `genomic-foundation-models, chromatin-state, intron-retention, epigenomics, sei-model`  
  — This study investigates intron retention (IR) as a regulated post-transcriptional event shaped not only by sequence motifs but also by chromatin context. The authors use Sei (a large-scale epigenomic foundation model) and DNABERT-2 to evaluate IR across tissues. Sei embeddings capture enhancer-like and splicing-regulatory chromatin states that strongly correlate with retained introns. Benchmarking shows Sei outperforms DNABERT-2 in predicting tissue-specific IR patterns. Attribution analyses reveal enrichment of chromatin accessibility, H3K36me3, and splicing-factor motifs near retained introns. The work highlights the necessity of multimodal models (sequence + epigenome) to decode RNA processing mechanisms.

- **"Are Genomic Language Models All You Need? Exploring Genomic Language Models on Protein Downstream Tasks"**, 2024  
  `genomic-language-models, cross-domain-transfer, nucleotide-transformer, protein-tasks`  
  — This cross-domain generalization study evaluates whether genomic foundation models can perform protein-level tasks. Using multiple Nucleotide Transformer variants (50M–2.5B parameters), the authors test performance on protein property prediction benchmarks and compare results to protein LLMs (ESM-2, ProtT5). Surprisingly, genomic models achieve competitive results on several tasks and even outperform protein LMs when trained with a new 3-mer tokenization scheme. A unified DNA–protein FM is introduced, exhibiting superior performance to single-domain models. The findings suggest shared representational structure across molecular modalities and motivate unified biological FMs.

- **"Large Language Models and Genomics for Summarizing the Role of MicroRNA in Regulating mRNA Expression"**, 2024  
  `mirna-mrna-interactions, llms, text-mining, literature-mining, genomics`  
  — This paper introduces the MMIC corpus, the first large-scale dataset for extracting microRNA–mRNA regulatory relationships from scientific literature. Classical ML models, PubMedBERT, and Llama-2 are benchmarked for information extraction. PubMedBERT achieves the highest F1 (0.783), while Llama-2 performs strongly in zero-shot settings with high recall. Error analysis shows difficulties in resolving ambiguous gene symbols, nested relations, and indirect regulatory statements. The study demonstrates how LLM-powered text mining supports automated curation of gene-regulatory networks.

- **"Single-Cell Foundation Models: Bringing Artificial Intelligence into Cell Biology"**, 2025  
  `single-cell-foundation-models, scgpt, geneformer, cell-type-annotation, perturbation-prediction`  
  — This survey reviews transformer-based foundation models for single-cell omics, trained on millions of scRNA-seq profiles across tissues and species. Models such as scGPT, Geneformer, scFoundation, and CellFM are compared across pretraining strategies (masked-gene modeling, graph-based attention, cell–gene bipartite transformers). Applications include automated cell-type annotation, developmental trajectory inference, batch correction, perturbation-response modeling, and regulatory network inference. Core challenges include extreme data sparsity, batch effects, cross-platform variability, and the need for biologically interpretable token embeddings. The review advocates for multimodal single-cell FMs integrating scATAC, proteomics, and spatial omics.

- **"Foundation Model: A New Era for Plant Single-Cell Genomics"**, 2025  
  `plant-single-cell, scplantllm, scgpt, geneformer, cross-species-mapping`  
  — This perspective highlights the emergence of plant-focused single-cell foundation models. The authors present scPlantLLM, trained on diverse plant scRNA-seq datasets to overcome challenges specific to plant biology, including polyploidy, rigid cell walls, and high transcriptional noise. The model performs zero-shot cell-type annotation, stress-response prediction, and integration of datasets from different tissues and species. Emphasis is placed on data scarcity, species divergence, and the need for plant-specific multimodal FMs integrating epigenomic and imaging data. The paper outlines a roadmap for cross-species generalization and plant developmental modeling.

- **"A Survey for Large Language Models in Biomedicine"**, 2024  
  `llms, biomedicine, multimodal-llms, zero-shot-evaluation, clinical-ai`  
  — A large-scale PRISMA-style survey covering 484 biomedical LLM publications. The review organizes applications into clinical NLP, diagnostic reasoning, biomedical Q&A, drug discovery, and precision medicine. Particular emphasis is placed on zero-shot and few-shot performance across 137 evaluation studies, revealing strong general linguistic reasoning but variable accuracy on specialized biomedical tasks. The authors categorize adaptation strategies (full fine-tuning, LoRA, prompting, retrieval-augmentation, multimodal alignment) and analyze limitations such as hallucinations, privacy constraints, non-transparent data sources, and lack of standardized evaluation. Future opportunities include federated training, clinically aligned benchmarks, model audits, and safety-aware deployment pipelines.

- **"Integrating Important Tile Filtering and Pathology Foundation Model for Lung Cancer Mutation Prediction"**, 2024  
  `pathology-foundation-models, lung-cancer, mutation-prediction, tile-filtering, wsi-analysis`  
  — This study proposes a two-stage pipeline for mutation prediction from whole-slide histopathology images (WSIs). First, important tile filtering selects discriminative tissue regions via k-means clustering and classifier-derived probabilities. Second, the pathology foundation model UNI extracts semantic features, which are fed into an InceptionResNet prediction head. The model achieves AUC ≈ 0.85 for TP53 mutations across TCGA and independent clinical cohorts, outperforming classical CNN and MIL baselines. Interpretability maps show phenotypic correlates of genetic mutations (e.g., keratinization patterns). The work highlights the power of foundation models for linking histopathological morphology to genomic alterations.

- **"Improvements in Viral Gene Annotation Using Large Language Models and Soft Alignments"**, 2024  
  `viral-genomics, protein-annotation, soft-alignment, embeddings, llms`  
  — The authors introduce a soft-alignment approach based on LLM-derived amino-acid embeddings to improve functional annotation of viral proteins, which often lack homology signals detectable by BLAST. The method compares embedding vectors position-wise to produce a similarity map visualized in a BLAST-like interface. Evaluations on Virus Orthologous Groups and ViralZone datasets show that the embedding-alignment method recovers remote homologs missed by blastp and pooled embedding baselines. The model significantly increases annotation coverage, offering a scalable strategy for characterizing orphan viral proteins.

- **"Artificial Intelligence–Assisted Breeding for Plant Disease Resistance"**, 2024  
  `plant-disease-resistance, ai-in-breeding, multimodal-models, phenomics, llms`  
  — This review summarizes AI methods for enhancing plant disease resistance, covering image-based disease detection, phenotypic trait extraction, genomics-guided QTL prediction, and multi-omics integration. It highlights how LLMs and multimodal foundation models can fuse genomic, transcriptomic, phenomic, and environmental data to accelerate breeding decisions. Challenges include limited annotated datasets, environmental variability, cross-species divergence, and the need for interpretable predictions. Future directions include federated learning for cross-lab collaboration, synthetic data via generative models, and LLM-based pipelines for large-scale plant breeding.

- **"A Conceptual Framework for Human–AI Collaborative Genome Annotation (HAICoGA)"**, 2024  
  `genome-annotation, llm-agents, human-ai-collaboration, explainable-ai, annotation-framework`  
  — HAICoGA proposes a hybrid genome-annotation ecosystem where automated tools and LLM-based agents collaborate with expert curators. The framework includes modules for evidence aggregation (from GENCODE, Ensembl, UniProt), gene-structure proposal, function summarization, variant significance reasoning, and prioritization of ambiguous loci. LLM agents generate candidate annotations and rationales, while experts validate outputs, ensuring traceability and interpretability. The paper identifies core needs: provenance-aware pipelines, uncertainty quantification, explainable rationales, and integration with experimental validation. HAICoGA outlines the next generation of semi-autonomous genome-annotation systems.

- **"The Application of Large Language Models to the Phenotype-Based Prioritization of Causative Genes in Rare Disease Patients"**, 2024  
  `rare-diseases, phenotype-analysis, gene-prioritization, hpo, llms`  
  — This study benchmarks GPT-3.5, GPT-4, and Falcon-180B for phenotype-driven gene prioritization using Human Phenotype Ontology (HPO) terms. Across real and synthetic cohorts, LLMs achieve competitive ranking performance when candidate-gene lists contain 5–100 genes, but remain inferior to specialized HPO-based tools. LLM predictions show biases toward well-known genes (BRCA1, PTEN) and occasional hallucinated gene–phenotype links. Free-text phenotype input improves performance relative to structured formats. The authors conclude that LLMs can provide rationales and hypothesis generation but should be used as complementary assistants rather than standalone diagnostic systems.

- **"A Visual–Omics Foundation Model to Bridge Histopathology with Spatial Transcriptomics (OmiCLIP + Loki)"**, 2024  
  `multimodal-foundation-models, spatial-transcriptomics, histopathology, clip-models, omics-integration`  
  — OmiCLIP employs dual encoders (H&E patches + gene-expression sentences) with a CLIP-style contrastive objective to learn joint visual–omics embeddings. Built on OmiCLIP, the Loki platform performs tissue alignment, spatial domain annotation, cross-modal retrieval, and prediction of spatial gene expression directly from histopathology. On multiple public datasets, Loki surpasses 22 state-of-the-art methods, offering robust generalization across tissues and technologies. Interpretability analyses link histologic regions to expression signatures such as immune infiltration and tumor-stroma interfaces. The approach demonstrates the feasibility of large-scale image–omics integration.

- **"Artificial Intelligence for Multiscale Spatial Analysis in Oncology"**, 2025  
  `multiscale-ai, oncology, radiomics, pathomics, spatial-omics, foundation-models`  
  — This review synthesizes AI approaches for integrating radiology (macro-scale), pathology (micro-scale), and spatial omics (molecular scale) to characterize tumor ecosystems. The authors discuss multiscale foundation models that align information across MRI/CT features, WSI histology, and spatial transcriptomics using cross-attention, graph fusion, and contrastive learning. Applications span tumor subtyping, treatment-response prediction, and microenvironment analysis. Key challenges include data heterogeneity, lack of standardized spatial benchmarks, biological interpretability, and the high computational cost of multiscale modeling. The paper proposes federated strategies and mechanistic AI to realize clinically actionable multiscale precision oncology.

- **"Emerging AI Approaches for Cancer Spatial Omics"**, 2024  
  `spatial-omics, cancer, ai-paradigms, mechanistic-modeling, foundation-models`  
  — This review categorizes spatial omics AI into three paradigms: (1) **data-driven foundation models** (transformers, GNNs, diffusion models), (2) **constraint-based AI** incorporating biological rules (e.g., reaction–diffusion constraints, interaction priors), and (3) **mechanistic spatial models** that explicitly simulate cell–cell interactions and nutrient diffusion. The paper evaluates spatial transcriptomics and proteomics methods for tumor-microenvironment analysis, highlighting interpretability and biological grounding as essential requirements. It calls for hybrid models combining large-scale pretrained embeddings with mechanistically consistent constraints to generate testable biological hypotheses.

- **"'Bingo': A Large Language Model– and Graph Neural Network–Based Workflow for Predicting Essential Genes from Protein Data"**, 2024  
  `protein-llms, esm2, essential-gene-prediction, gnn, explainable-ai`  
  — Bingo integrates ESM-2 protein language model embeddings with a graph neural network classifier to identify essential genes across metazoans using only protein sequences. The pipeline employs adversarial training for robustness and GNNExplainer for motif- and domain-level interpretability. Benchmarks in *C. elegans*, *D. melanogaster*, mouse, and human (HepG2) demonstrate high accuracy and strong zero-shot transfer to under-annotated species. Insights reveal conserved structural motifs and domains linked to essentiality, outperforming traditional ML and GNN approaches. The study showcases the power of protein LLMs for cross-species functional genomics.

- **"Biomedical Information Integration via Adaptive Large Language Model Construction (TSLLM)"**, 2025, IEEE JBHI  
  `biomedical-entity-alignment, llm-ensemble, genetic-programming, mogp, soga`  
  — TSLLM is a two-stage system for biomedical entity alignment across heterogeneous ontologies. Stage (1): Multi-Objective Genetic Programming (MOGP) constructs a diverse population of LLM-based matchers, each capturing different embedding and prompting strategies. Stage (2): a Single-Objective Genetic Algorithm (SOGA) ensembles the best-performing matchers using learned confidence weights. The framework is evaluated on OAEI Benchmark datasets (Conference, LargeBio, Disease, Phenotype), achieving significant improvements over classical and neural baselines. A new “expert-free quality metric” is proposed for evaluating biomedical alignment without human gold standards. TSLLM demonstrates how evolutionary optimization can create robust LLM ensembles for biomedical knowledge integration.

- **"ViraLM: Empowering Virus Discovery through the Genome Foundation Model"**, 2025, Bioinformatics  
  `viral-detection, dnabert-2, genome-foundation-models, metagenomics, short-contigs`  
  — ViraLM fine-tunes DNABERT-2 on 49,929 curated viral genomes alongside challenging nonviral sequences (bacterial, archaeal, and eukaryotic host genomes). The model learns discriminative genomic-signature embeddings optimized for identifying viral contigs—especially very short ones (<1 kb), which are typically hard to classify. Benchmarks on RefSeq, IMG/VR, and real metagenomic studies show ViraLM outperforming geNomad, VIRify, VIBRANT, and DeepVirFinder with up to +22% F1 improvement on short contigs. ViraLM also identifies novel viral sequences missed by protein-based tools, demonstrating the value of DNA-level LLMs for large-scale viral discovery.

- **"DruGNNosis-MoA: Drug Mechanisms as Etiological or Palliative with Graph Neural Networks Employing a Large Language Model"**, 2025, IEEE JBHI  
  `drug-mechanisms, scibert, gnn, drug-repurposing, mechanism-of-action`  
  — DruGNNosis-MoA integrates SciBERT embeddings of drug descriptions with a graph neural network (GNN) representing drug–gene–disease interactions. The model classifies each drug’s mechanism of action (MoA) as “etiological” (disease-causal targeting) or “palliative” (symptom control). Evaluated on 2,018 FDA-approved drugs, the hybrid LLM–GNN achieves F1 ≈ 0.94, surpassing SciBERT-only and GNN-only baselines. Interpretability via attention and node-importance maps highlights pathways and protein modules that drive MoA classification. The method supports systematic drug repurposing and precision-medicine decisions.

- **"Deep Learning for Genomics: From Early Neural Nets to Modern Large Language Models"**, 2025  
  `genomics-review, deep-learning, dna-llms, regulatory-genomics, multimodal-models`  
  — This comprehensive review traces the evolution of deep learning in genomics: from early CNN/RNN models to modern transformer-based genomic foundation models such as Enformer, Nucleotide Transformer, DNABERT, and HyenaDNA. Key application domains include variant effect prediction, chromatin accessibility modeling, enhancer annotation, and 3D genome structure inference. The review outlines challenges for genomic LLMs, including long-range dependency capture, limited labeled data, and interpretability barriers. Future directions focus on multimodal integration (genome + epigenome + transcriptome), sparse attention for scaling, and biologically grounded explainability frameworks.

- **"Developing a Predictive Platform for Salmonella Antimicrobial Resistance Based on a Large Language Model and Quantum Computing (SARPLLM)"**, 2025  
  `antimicrobial-resistance, salmonella, llm-adaptation, quantum-computing, pan-genomics`  
  — SARPLLM predicts antimicrobial resistance (AMR) in *Salmonella* using pan-genomic features. A two-step feature-selection pipeline combines chi-square filtering with conditional mutual information maximization, identifying key AMR-associated genes. A Qwen2 LLM fine-tuned with LoRA performs the final AMR classification. A quantum-inspired augmentation algorithm (QSMOTEN) reduces computational complexity by compressing nearest-neighbor distance calculations from O(n) to O(log n). The system includes an online visualization platform integrating knowledge graphs, pan-genome analytics, and real-time AMR prediction. SARPLLM highlights the synergy between classical genomics, LLM adaptation, and quantum-inspired computation.

- **"Advancing Plant Single-Cell Genomics with Foundation Models"**, 2025  
  `plant-single-cell, foundation-models, generative-models, scRNA-seq, multimodal-ai`  
  — This review surveys the rise of foundation models in plant single-cell genomics. Plant scRNA-seq suffers from challenges such as rigid cell walls, sparsity, species divergence, and limited datasets. The authors examine Transformer-based models (scGPT, Geneformer, scFoundation) and how they adapt to plant biology, supporting tasks such as cell-type annotation, gene-network modeling, and stress-response mapping. The paper also covers generative models (GANs, diffusion) for synthetic single-cell data generation to address data scarcity. Future directions include multimodal integration (scATAC + imaging), species-aware tokenization, and scalable cross-species embedding transfer for crop improvement.

- **"An AI Agent for Fully Automated Multi-Omic Analyses (AutoBA)"**, 2024  
  `llm-agents, automated-analysis, rna-seq, chip-seq, spatial-transcriptomics, code-generation`  
  — AutoBA is an autonomous LLM-based agent capable of end-to-end analysis of multi-omics data (WGS, WES, RNA-seq, scRNA-seq, ATAC-seq, ChIP-seq, ST). With only three user inputs (data path, description, final objective), the agent designs pipelines, generates code, executes it, and automatically repairs errors using an Automated Code Repair (ACR) module. AutoBA integrates mainstream bioinformatics tools (FastQC, HISAT2/STAR, BWA, Salmon/Kallisto, MACS2, DESeq2/EdgeR, Seurat, ChIPseeker). Evaluations show high stability, reproducibility, and adaptability across tasks. The framework demonstrates the potential of agentic LLMs for low-code biomedical data analysis.

- **"XMolCap: Advancing Molecular Captioning Through Multimodal Fusion and Explainable Graph Neural Networks"**, 2025  
  `molecular-captioning, multimodal-fusion, gnn, llms, drug-discovery`  
  — XMolCap combines SMILES/SELFIES strings, molecular images, and graph-based features to generate chemically accurate natural-language captions. Built on a BioT5 encoder–decoder backbone, the system integrates SwinOCSR for molecular-image OCR, SciBERT for textual chemical understanding, and GIN-MoMu for structural graph encoding. The stacked fusion mechanism jointly aligns visual, textual, and graph modalities. On L+M-24 and ChEBI-20 benchmarks, XMolCap achieves state-of-the-art captioning accuracy and provides interpretable GNN-based explanations highlighting functional groups and reactive substructures. Applications include drug design, molecular education, and compound database annotation.

- **"The Rise and Potential Opportunities of Large Language Model Agents in Bioinformatics and Biomedicine"**, 2025  
  `llm-agents, autonomous-ai, multiagent-systems, drug-discovery, clinical-ai`  
  — This review analyzes LLM agents—systems that integrate LLM reasoning with planning, tool use, memory, and multi-agent coordination. Applications span multi-omics analysis, drug discovery, literature mining, laboratory automation, and patient-management workflows. Architectural components include planners, tool APIs, vector-memory stores, world models, and agent-to-agent communication. Key challenges include hallucinations during tool invocation, privacy/security concerns, temporal drift in agent memory, and lack of benchmarks for agentic evaluation. The paper envisions “AI scientist” systems capable of hypothesis generation, experiment design, and semi-autonomous biomedical research.

- **"AuraGenome: An LLM-Powered Framework for On-the-Fly Reusable and Scalable Circular Genome Visualizations"**, 2025  
  `genome-visualization, llm-agents, circular-genomics, d3-visualization`  
  — AuraGenome is an LLM-driven multiagent system that generates, edits, and scales circular genome visualizations (e.g., ring, radial, and chord diagrams) without manual coding. Seven specialized agents handle intent parsing, layout design, data parsing, D3.js code generation, validation, and explanation. A layer-aware reuse mechanism enables users to repurpose visualization components across multiple datasets. User studies show substantial improvements in speed, correctness, and usability compared to tools like Circos. AuraGenome demonstrates how LLM agents can democratize complex bioinformatics visualization tasks.

- **"Assessing the Utility of Large Language Models for Phenotype-Driven Gene Prioritization in the Diagnosis of Rare Genetic Disease"**, 2024  
  `rare-disease-diagnostics, llms, gene-prioritization, hpo, clinical-ai`  
  — This benchmark evaluates five LLMs (GPT-4, GPT-3.5, three Llama-2 variants) on phenotype-driven gene prioritization using HPO terms. Accuracy for ranking the causal gene within the top 50 is ~17% for GPT-4—still below knowledge-graph–based tools. Prompt variations (structured HPO, free-text, case summaries) reveal LLMs perform better with narrative input. RAG and few-shot prompting do not significantly improve results. Models display citation bias toward well-known disease genes (e.g., BRCA1, PTEN). Despite low diagnostic accuracy, LLMs generate interpretable rationales and phenotype–variant hypotheses, positioning them as assistants in diagnostic pipelines rather than replacements.

- **"Utilizing Omic Data to Understand Integrative Physiology"**, 2024  
  `multi-omics, integrative-physiology, nlp-in-biology, bayesian-inference`  
  — This review discusses how multi-omics (transcriptomics, proteomics, metabolomics) contributes to integrative physiology research, but also highlights limitations in reconstructing organism-level biological functions from reductionist datasets. The authors identify three progress areas: (1) user-friendly, cross-indexed omics databases, (2) Bayesian frameworks combining multi-omics evidence with physiological priors, and (3) NLP/LLM systems that mine literature to build causal networks of physiological mechanisms. The review emphasizes LLM limitations, particularly difficulty in integrating structured omics data, dealing with causality, and avoiding hallucinated mechanistic links. Future directions include multimodal FMs and causal-inference–aware LLMs.

- **"Improvements in Viral Gene Annotation Using Large Language Models and Soft Alignments"**, 2024  
  `viral-genomics, llms, soft-alignment, remote-homology, protein-annotation`  
  — The paper introduces a soft alignment framework using LLM-based amino-acid embeddings to overcome the limitations of BLAST-like methods for remote viral protein homology. The approach aligns proteins via learned embedding similarity, generating an interpretable BLAST-like heatmap. Benchmarks on Virus Orthologous Groups and ViralZone show the method uncovers remote homologs missed by blastp and pooled-embedding baselines. Significant annotation improvements are reported for structurally divergent viral proteins. The approach offers a scalable strategy for improving viral gene annotation pipelines and identifying orphan proteins.

- **"Artificial Intelligence–Assisted Breeding for Plant Disease Resistance"**, 2024  
  `plant-disease-resistance, phenomics, llms, multimodal-models, genomic-selection`  
  — This review highlights AI-driven strategies for plant disease resistance, including phenomics-based image classifiers, multi-omics integration for quantitative trait modeling, and genomic selection using machine learning. The authors discuss how LLMs and foundation models can unify genomic, transcriptomic, phenotypic, and environmental variables, enabling prediction of resistance traits and faster breeding cycles. Limitations include sparse plant omics datasets, strong environmental effects, limited interpretability, and low cross-species transfer. Future visions include LLM-driven collaborative breeding platforms, federated training across field stations, and synthetic data augmentation via generative models.

- **"A Conceptual Framework for Human–AI Collaborative Genome Annotation (HAICoGA)"**, 2024  
  `genome-annotation, llm-agents, explainable-ai, interactive-ml`  
  — HAICoGA outlines a human–AI collaboration paradigm for scalable genome annotation. LLM agents propose gene structures, function summaries, and variant interpretations by aggregating evidence from databases such as Ensembl, GENCODE, and UniProt. Human experts validate and refine the suggestions, forming an iterative feedback loop. The framework emphasizes provenance tracking, rationales with uncertainty estimates, and interpretable decision-making. HAICoGA addresses challenges in data fragmentation, limited experimental annotations, and lack of transparency in current annotation tools, representing a pathway toward semi-autonomous, expert-supervised LLM annotation systems.

- **"A Visual–Omics Foundation Model to Bridge Histopathology with Spatial Transcriptomics (OmiCLIP + Loki)"**, 2024  
  `visual-omics, clip-models, spatial-transcriptomics, histopathology, multimodal-foundation-models`  
  — OmiCLIP introduces dual encoders—one for H&E histopathology patches and one for spatial transcriptomics (ST) gene-expression sentences—trained using a CLIP-style contrastive objective. The follow-up platform Loki leverages these embeddings for tissue alignment, spatial domain segmentation, and cross-modal retrieval. Loki can also predict gene expression directly from histology images. Evaluated across large public and in-house ST datasets, OmiCLIP/Loki outperforms 22 competing models in clustering accuracy, gene-expression prediction (PCC), and cell-type domain resolution. Interpretability analysis shows alignment between histologic morphology and transcriptomic signatures, enabling cross-modality biological insights.

- **"Artificial Intelligence for Multiscale Spatial Analysis in Oncology"**, 2025  
  `multiscale-ai, oncology, radiomics, pathomics, spatial-omics, tumor-microenvironment`  
  — This review integrates AI advances across three spatial scales in oncology: radiology (macro), histopathology (micro), and spatial omics (molecular). The paper discusses multiscale foundation models that align information across MRI/CT radiomics, WSI pathomics, and ST/IMC spatial omics through cross-attention, graph fusion, and multimodal contrastive learning. Applications include tumor subtyping, immune-microenvironment characterization, and therapy-response prediction. Core challenges include cross-platform heterogeneity, lack of unified annotation standards, computational cost, and biological interpretability. The authors propose mechanistic AI and federated training pipelines for clinically deployable multiscale oncology systems.

- **"Emerging AI Approaches for Cancer Spatial Omics"**, 2024  
  `spatial-omics, cancer, foundation-models, mechanistic-modeling, interpretable-ai`  
  — This review categorizes spatial-omics AI into three paradigms: (1) data-driven foundation models (transformers, GNNs, VAEs, diffusion models), (2) knowledge/constraint-based AI integrating biological priors such as ligand–receptor interactions and reaction–diffusion physics, and (3) mechanistic spatial models simulating cell–cell interactions and nutrient gradients. The authors summarize how each paradigm is applied to spatial transcriptomics and proteomics for tumor-microenvironment analysis. A key theme is interpretability—linking spatial gene-expression patterns to biological processes. They advocate for hybrid mechanistic–data-driven AI capable of generating testable hypotheses.

- **"'Bingo': A Large Language Model– and Graph Neural Network–Based Workflow for Predicting Essential Genes from Protein Data"**, 2024  
  `essential-genes, protein-llms, gnn, esm2, explainable-ai`  
  — Bingo integrates ESM-2 protein embeddings with a graph neural network classifier to predict essential genes using only protein sequences. The workflow includes adversarial training for robustness and GNNExplainer for structural/motif-level biological interpretability. The model achieves high cross-species accuracy in *C. elegans*, *D. melanogaster*, mouse, and human (HepG2), with strong zero-shot performance on poorly annotated species. Interpretability analyses highlight conserved protein domains and motifs associated with essentiality. Bingo outperforms classical ML and standalone GNN methods, demonstrating the potential of protein LLMs for functional genomics.

- **"Biomedical Information Integration via Adaptive Large Language Model Construction (TSLLM)"**, 2025, IEEE JBHI  
  `entity-alignment, llm-ensembles, evolutionary-search, mogp, soga`  
  — TSLLM constructs adaptive LLM ensembles for biomedical entity-alignment tasks. Multi-Objective Genetic Programming (MOGP) evolves diverse LLM-based matchers (varying prompts, embeddings, and similarity metrics), while a Single-Objective Genetic Algorithm (SOGA) determines optimal ensembling weights. The framework is evaluated on OAEI datasets—Conference, LargeBio, Disease, Phenotype—achieving superior matching precision and recall over rule-based and deep-learning baselines. TSLLM introduces an “expert-free” metric enabling automatic evaluation of entity-alignment quality without ground-truth labels. This work highlights how evolutionary strategies can optimize LLM integration in biomedical knowledge graphs.

- **"ViraLM: Empowering Virus Discovery through the Genome Foundation Model"**, 2025, Bioinformatics  
  `viral-discovery, genome-foundation-models, dnabert-2, metagenomics, short-contigs`  
  — ViraLM is a viral-detection framework built on DNABERT-2 and trained on nearly 50,000 curated viral genomes plus challenging nonviral sequences. The model learns discriminative genomic signatures enabling detection of viral contigs — especially ultrashort (<1 kb) and low-diversity sequences common in metagenomic data. Benchmarks on RefSeq, IMG/VR, and real metagenomes show up to +22% F1 improvement over protein-based tools (geNomad, VIBRANT, VirSorter2) and DNA-based baselines (VirRep, DeepVirFinder). ViraLM also discovers novel viral candidates missed by protein-homology workflows. Its robustness across hosts and sequencing platforms demonstrates the power of DNA-level LLMs for scalable viral surveillance.

- **"DruGNNosis-MoA: Drug Mechanisms as Etiological or Palliative With Graph Neural Networks Employing a Large Language Model"**, 2025, IEEE JBHI  
  `drug-mechanisms, scibert, gnn, drug-repurposing, mechanistic-interpretation`  
  — This study integrates SciBERT embeddings with a drug–gene–disease graph neural network to classify drugs as “etiological” (targeting root causes) or “palliative” (modifying symptoms). Three methodological variants are compared: SciBERT alone, GNN alone, and the integrated LLM–GNN model. The hybrid model achieves the best F1-score (~0.94) across 2,018 FDA-approved drugs. Interpretability tools (attention maps, node-importance scores) highlight mechanistic pathways that drive classification decisions. The work demonstrates the utility of foundation-model embeddings combined with biological knowledge graphs for explainable MoA characterization and precision-medicine repurposing.

- **"Deep Learning for Genomics: From Early Neural Nets to Modern Large Language Models"**, 2025  
  `genomics-review, dna-llms, enformer, nucleotide-transformer, multimodal-genomics`  
  — This review traces 30+ years of genomic deep learning, progressing from early multilayer perceptrons through CNN/RNN architectures to present-day genomic foundation models such as Enformer, Nucleotide Transformer, DNABERT, HyenaDNA, Caduceus, and MambaDNA. Key application domains include variant-effect prediction, enhancer/chromatin modeling, promoter classification, and 3D genome inference. The survey highlights core challenges: modeling long-range dependencies, multi-omics integration, interpretability, label scarcity, and sequencing biases. Future directions include multimodal genomic FMs (DNA + epigenomics + transcriptomics), sparse/linear attention for 1M+ token contexts, large cross-species pretraining, and experimentally guided interpretability for regulatory genomics.

- **"Advancing Plant Single-Cell Genomics with Foundation Models"**, 2025  
  `single-cell-genomics, foundation-models, plant-genomics, generative-models, multimodal-omics`  
  — This review analyzes how foundation models (FMs) and deep-learning architectures are transforming plant single-cell genomics. Plant scRNA-seq poses challenges including cellular dissociation difficulty, extreme sparsity, species divergence, and limited dataset availability. The authors examine transformer-based FMs such as GPT-like autoregressive models, BERT-style masked language models, and specialized architectures (scGPT, Geneformer, scFoundation), highlighting how these models improve cell-type annotation, developmental trajectory inference, and gene regulatory network modeling.  
  — The paper emphasizes multimodal integration, demonstrating how FMs combine scRNA-seq with scATAC-seq, proteomics, and spatial transcriptomics to build unified, biologically meaningful embeddings. A major focus is placed on generative approaches—GANs and diffusion models—which enable high-fidelity synthetic plant single-cell data generation, reduce dropout artifacts, and address class imbalance in rare cell populations.  
  — The review also introduces plant-focused foundation models such as scPlantLLM, designed for polyploid genomes, species-specific gene families, and cross-species transfer. scPlantLLM demonstrates strong zero-shot cell-type annotation and stress-response prediction across diverse plant species. The authors outline future directions including multimodal plant FMs, species-aware tokenization, and large-scale cross-species pretraining to accelerate discoveries in plant development, stress resilience, and crop improvement.

- **"An AI Agent for Fully Automated Multi-Omic Analyses (AutoBA)"**, 2024  
  `llm-agents, multi-omics, automated-analysis, code-generation, bioinformatics-pipelines`  
  — AutoBA is an autonomous LLM-driven agent designed to perform fully automated multi-omic bioinformatics analyses with minimal user input. The system requires only three inputs—data path, data description, and analysis objective—and then automatically generates analysis plans, writes code, executes pipelines, debugs errors, and performs downstream interpretation. Built to address challenges in bioinformatics reproducibility, pipeline variability, and the high training cost for wet-lab researchers, AutoBA integrates multiple LLM backends (online and local) to ensure data security and privacy. Its Automated Code Repair (ACR) mechanism enhances reliability by identifying and fixing execution failures during pipeline generation. AutoBA supports a wide range of omics data types including WGS, WES, RNA-seq, scRNA-seq, ATAC-seq, ChIP-seq, and spatial transcriptomics, and leverages mainstream tools such as FastQC, Trimmomatic, HISAT2/STAR/BWA, Salmon/Kallisto, MACS2, DESeq2/EdgeR, Seurat, and ChIPseeker. Benchmarks across diverse real-world multi-omic datasets show strong stability, adaptability, and performance exceeding general-purpose LLMs (e.g., ChatGPT) and online analysis services. AutoBA represents a major advancement in LLM-based autonomous bioinformatics, enabling low-code, reproducible, and scalable end-to-end analyses adaptable to emerging tools and methodologies.

- **"XMolCap: Advancing Molecular Captioning Through Multimodal Fusion and Explainable Graph Neural Networks"**, 2025  
  `molecular-captioning, multimodal-fusion, graph-neural-networks, llms, drug-discovery`  
  — XMolCap introduces a multimodal, explainable molecular captioning framework that integrates three complementary molecular representations: molecular images, SMILES/SELFIES strings, and graph-based structural information. Built on a BioT5 encoder–decoder backbone, the system leverages specialized modules including SwinOCSR for molecular-image OCR, SciBERT for chemical language understanding, and GIN-MoMu for graph-based structural encoding. A stacked multimodal fusion mechanism jointly aligns visual, textual, and graph embeddings to generate accurate and chemically grounded natural-language captions. XMolCap achieves state-of-the-art performance on two benchmark datasets (L+M-24 and ChEBI-20), outperforming several strong baselines. The framework provides interpretable, functional group–aware explanations via graph-based attention, highlighting key substructures and molecular properties that influence caption generation. The tool is publicly available for reproducibility and supports local deployment. XMolCap demonstrates the potential of multimodal LLM–GNN integration for interpretable molecular representation learning, with applications in drug discovery and chemical informatics.

- **"The Rise and Potential Opportunities of Large Language Model Agents in Bioinformatics and Biomedicine"**, 2025  
  `llm-agents, autonomous-ai, multi-agent-systems, bioinformatics, biomedicine`  
  — This review provides a comprehensive overview of LLM agents—systems that extend large language models with capabilities for reasoning, planning, tool use, and autonomous task execution. The authors outline the technical foundations of LLM agents, including core architectural components (planners, memory modules, tool APIs, environment interfaces), multi-agent collaboration modes, and enabling technologies such as retrieval augmentation, function calling, and tool orchestration. The paper examines diverse applications across bioinformatics and biomedicine, including automated multi-omics analysis, drug discovery pipelines, chemical reasoning, clinical decision support, diagnostic triage, and personalized health management. It also highlights the limitations of current LLM agents: framework scalability, tool orchestration complexity, privacy and data security concerns, model hallucinations, interpretability challenges, lack of real-time knowledge updates, and ethical/regulatory risks in clinical settings. Future directions include standardized open-source ecosystems for biomedical agents, human–AI collaborative paradigms, secure agent frameworks, continuous knowledge-refresh pipelines, and the emergence of “AI scientist” multi-agent systems capable of hypothesis generation, experiment planning, and autonomous biomedical reasoning. This review positions LLM agents as a transformative next step beyond conventional LLMs, with significant potential to accelerate precision medicine, biomedical research, and computational biology.

- **"AuraGenome: An LLM-Powered Framework for On-the-Fly Reusable and Scalable Circular Genome Visualizations"**, 2025  
  `genome-visualization, llm-agents, circular-genomics, visualization-frameworks, multiagent-systems`  
  — AuraGenome introduces an LLM-driven multiagent framework for generating reusable, scalable circular genome visualizations without manual scripting. The system addresses limitations of traditional tools such as Circos, which require complex configuration and iterative parameter tuning. AuraGenome employs seven specialized LLM-powered agents responsible for intent recognition, semantic parsing, layout design, D3.js code generation, validation, refinement, and explanation, enabling natural-language-driven visualization creation. Built atop this semantic multiagent workflow, the interactive visual analytics system supports multilayered circular layouts—including ring, radial, and chord representations—to visualize genomic features such as structural variants, chromosomal interactions, and regulatory elements. A layer-aware reuse mechanism allows users to adapt and repurpose visualization components across tasks, improving efficiency and narrative report generation. Validation across two real-world case studies and a comprehensive user study demonstrates significant improvements in usability, speed, and flexibility compared to traditional circular-genomics visualization pipelines. AuraGenome highlights the potential of LLM agents to democratize complex genomic visualization tasks by combining natural-language interfaces with automated, high-quality D3-based visualization generation.

- **"Assessing the Utility of Large Language Models for Phenotype-Driven Gene Prioritization in the Diagnosis of Rare Genetic Disease"**, 2025  
  `rare-disease-diagnosis, gene-prioritization, hpo, llms, clinical-genomics`  
  — This study systematically evaluates five LLMs—GPT-3.5, GPT-4, Llama2-7B, Llama2-13B, and Llama2-70B—for phenotype-driven gene prioritization, a core task in diagnosing rare genetic disorders. The authors assess performance across task completeness, gene ranking accuracy, and adherence to required output structures using multiple prompting strategies, phenotypic input formats (free text vs. standardized HPO concepts), and task difficulty levels. Despite improvements with larger models and more sophisticated prompts, GPT-4 achieves only 17% accuracy in identifying the causal gene within the top 50 predictions—substantially lower than traditional tools such as Phenomizer, Exomiser, AMELIE, and Phen2Gene, which rely on curated phenotype–gene knowledge graphs. The study finds that free-text input yields better-than-random predictions but remains slightly inferior to structured HPO input. Neither retrieval-augmented generation (RAG) nor few-shot prompting improves performance, and complex prompts reduce output-structure compliance. Bias analyses reveal a strong preference for highly studied genes (e.g., BRCA1, TP53, PTEN). Using a post-2023 dataset confirms the robustness of findings and reduces concerns about training-data leakage. The study concludes that while LLMs generate coherent rationales and can support hypothesis generation, they are not yet reliable replacements for dedicated phenotype-gene prioritization tools in clinical genomics.

- **"Utilizing Omic Data to Understand Integrative Physiology"**, 2024  
  `multi-omics, integrative-physiology, nlp-in-biology, bayesian-inference, llms`  
  — This review analyzes the challenges and opportunities in integrating large-scale omic data—particularly protein mass spectrometry and next-generation sequencing—with traditional hypothesis-driven physiology to understand organism-level biological mechanisms. The author summarizes key omic techniques relevant to physiological research and highlights three major advancements enabling integrative physiology: (1) development of cross-indexed, user-friendly omic databases that democratize access and unify heterogeneous datasets; (2) application of Bayesian frameworks to combine multi-omics evidence with mechanistic insights from classical physiology, enabling probabilistic reasoning about biological processes; and (3) use of natural language processing to mine literature and construct causal graphs that represent physiological pathways in a structured, machine-readable form. The review discusses emerging applications of large language models, including their potential to support literature synthesis and hypothesis generation, but also emphasizes current limitations such as hallucination, inability to integrate structured physiological datasets, and challenges in generating mechanistically accurate causal inferences. Overall, the paper provides a roadmap for combining omics, computational inference, and NLP/LLM technologies to advance whole-organism physiological understanding.

- **"Nicheformer: A Foundation Model for Single-Cell and Spatial Omics"**, 2025  
  `single-cell-foundation-models, spatial-omics, transformer-models, multimodal-pretraining, spatial-transcriptomics`  
  — Nicheformer is a transformer-based foundation model trained jointly on dissociated single-cell and spatial transcriptomics data to learn spatially informed cellular representations at scale. The authors curate SpatialCorpus-110M, a massive dataset containing over 110 million human and mouse cells—including 57M dissociated and 53M spatially resolved cells across 73 tissues—and pretrain Nicheformer using self-supervision to integrate both molecular expression and spatial context. By incorporating modality, organism, and assay tokens, the model learns unified representations that encode microenvironmental structure. Nicheformer achieves strong performance in linear probing and fine-tuning across newly designed downstream tasks, particularly in spatial composition prediction and spatial label prediction, outperforming existing foundation models such as Geneformer, scGPT, UCE, CellPLM, scVI, and PCA-based embeddings. Importantly, models trained only on dissociated data fail to capture microenvironmental complexity, underscoring the need for spatial-aware pretraining. Nicheformer enables accurate inference of spatial context for dissociated scRNA-seq datasets, effectively transferring spatial information from spatial transcriptomics data. This work establishes a next-generation foundation model paradigm for robust spatially informed representation learning in single-cell biology.

- **"Steering Veridical Large Language Model Analyses by Correcting and Enriching Generated Database Queries: First Steps Toward ChatGPT Bioinformatics"**, 2025  
  `llm-accuracy, bioinformatics-assistants, database-query-correction, rag, llm-steering`  
  — This study examines the limitations of ChatGPT as a bioinformatics assistant, revealing consistent problems in data retrieval, silent hallucinations, incorrect sequence manipulations, API misuse, and flawed code generation. To address these issues, the authors introduce **NagGPT**, a middleware system placed between LLMs and genomics databases that intercepts, corrects, and enriches LLM-generated queries. NagGPT validates and fixes malformed database requests, synthesizes large responses into concise snippets, and injects corrective comments back into the LLM prompt to steer reasoning. A companion custom GPT—**GenomicsFetcher-Analyzer (GFA)**—integrates ChatGPT with NagGPT, enabling dynamic retrieval of authoritative data from major genomics resources (NCBI, Ensembl, UniProt, WormBase, FlyBase) and execution of real bioinformatics software through generated Python code. Despite partial mitigations—including handling identifier confusion, improving API call consistency, and reducing hallucinated operations—the authors find that silent errors still occur, requiring user oversight and manual debugging. The work highlights significant challenges in using unmodified LLMs for scientific workflows but demonstrates a viable path toward veridical, tool-augmented LLM systems for future bioinformatics assistants.

- **"FHG-GAN: Fuzzy Hypergraph Generative Adversarial Network With Large Foundation Models for Alzheimer’s Disease Risk Prediction"**, 2025  
  `alzheimer-risk-prediction, multiomics, fuzzy-hypergraphs, generative-adversarial-networks, foundation-models`  
  — FHG-GAN is a fuzzy hypergraph–based deep learning framework designed to integrate multiomics data for Alzheimer’s disease (AD) risk prediction and mechanistic insight. The method begins by formulating a mathematical model of fuzzy structural entropy propagation, representing AD progression as topological evolution within fuzzy hypergraphs that encode high-order, uncertain associations among brain regions and genes. Large foundation models (BrainLM for fMRI and Nucleotide Transformer for SNPs) generate high-quality feature embeddings, mitigating noise and inconsistencies in heterogeneous biomedical data. These embeddings are used to construct brain-region–gene fuzzy hypergraphs. The proposed FHG-GAN employs fuzzy hypergraph convolutional layers within its generator to model disease evolution patterns, while the discriminator assesses real versus generated hypergraphs. Across multiple datasets, FHG-GAN outperforms advanced baselines in AD risk prediction, multiomics feature fusion, and evolutionary pattern discovery. It accurately extracts pathogenic brain lesions and risk genes, offering interpretable insights into disease mechanisms and supporting earlier diagnosis of AD-like neurodegenerative disorders.

- **"A Foundational Large Language Model for Edible Plant Genomes (AgroNT)"**, 2025  
  `plant-genomics, dna-foundation-models, crop-improvement, regulatory-prediction, zero-shot-variant-scoring`  
  — AgroNT is a transformer-based foundational DNA language model trained on reference genomes from 48 plant species, with a strong emphasis on edible and agriculturally significant crops. The model leverages large-scale self-supervised training to learn sequence representations directly from genomic DNA, enabling accurate prediction of diverse regulatory and functional genomic features without relying on extensive labeled datasets. AgroNT achieves state-of-the-art performance across tasks including regulatory element annotation, promoter/terminator strength prediction, tissue-specific gene expression inference, and functional variant prioritization. Demonstrating strong zero-shot capabilities, the model accurately predicts the impact of variants even in understudied “orphan crops.” The authors perform large-scale in silico saturation mutagenesis on cassava, evaluating over 10 million mutations to map their regulatory effects, providing a valuable public resource for variant characterization. AgroNT is released on HuggingFace, and the study introduces the Plants Genomic Benchmark (PGB), a comprehensive multi-task benchmark for evaluating deep-learning approaches in plant genomics. The results highlight AgroNT’s utility for advancing crop genomic improvement, regulatory annotation, and model-guided genome editing across diverse plant species.

- **"Optimized Biomedical Entity Relation Extraction with Data Augmentation and Classification Using GPT-4 and Gemini"**, 2024  
  `biomedical-ner, relation-extraction, llm-augmentation, gpt4, gemini, bionlp`  
  — This work proposes a hybrid large-language-model–enhanced pipeline for biomedical named entity recognition (NER) and relation extraction (RE), addressing challenges such as multistage prediction, ontology-dependent entity identifiers, unbalanced datasets, and cross-sentence relations. The approach integrates GPT-4 for synthetic data augmentation, Gemini for generating enriched relation-aware outputs, and an ensemble of fine-tuned BioNLP–PubMedBERT classifiers for final prediction. The system is designed to overcome the limitations of existing models: AIONER lacks identifier normalization, BERT-GT does not perform NER, and BioREX omits novelty prediction. Leveraging LLM-generated augmented data improves robustness to rare relation types and enhances coverage beyond sentence-level interactions. Experimental results on the **BioCreative VIII BioRED** benchmark show consistent gains in precision, recall, and F1, including improved detection of relation types (“binding,” “association,” “drug interaction,” etc.) and prediction of the “novelty” attribute. The LLM-augmented framework demonstrates that GPT-4 and Gemini can meaningfully enhance RE performance when combined with domain fine-tuning, but still require curated classification models for reliable biomedical extraction.

- **"Leveraging Large Language Models to Predict Antibiotic Resistance in Mycobacterium tuberculosis (LLMTB)"**, 2025  
  `antimicrobial-resistance, mtb-genomics, llms, transformer-models, antibiotic-resistance-prediction`  
  — LLMTB introduces a transformer-based large language model for predicting antibiotic resistance in *Mycobacterium tuberculosis* (MTB), trained on genomic data from 12,185 CRyPTIC isolates and evaluated on an independent set of 5,954 isolates. Motivated by the limitations of culture-based susceptibility testing and curated mutation-based tools (TBProfiler, Mykrobe, ResFinder, KvarQ), LLMTB leverages BERT-style architectures to extract genomic resistance signals directly from sequences without relying on predefined resistance variant lists. The model employs gene-level tokenization to capture biologically meaningful patterns and enhance interpretability. LLMTB achieves high predictive performance across 13 antibiotics—often matching or surpassing traditional AMR tools—while enabling fine-tuning and few-shot learning to rapidly adapt to emerging drugs. Attention analyses highlight relevant genes and intergenic regions, revealing known and potentially novel resistance determinants. Beyond accurate AMR classification, LLMTB offers deeper biological insights and demonstrates the potential of LLMs to generalize across genomic contexts, supporting improved diagnostics and personalized treatment strategies for drug-resistant TB.

- **"Large Language Models Facilitate the Generation of Electronic Health Record Phenotyping Algorithms"**, 2024  
  `ehr-phenotyping, clinical-informatics, llms, sql-generation, structured-health-data`  
  — This study evaluates the ability of four LLMs—GPT-4, GPT-3.5, Claude 2, and Bard—to generate computable electronic health record (EHR) phenotyping algorithms expressed as SQL queries aligned with a common data model (CDM). Phenotyping experts assessed LLM-generated algorithms for three clinical conditions (type 2 diabetes, dementia, and hypothyroidism) across instruction following, concept identification, logic accuracy, and executability. GPT-4 and GPT-3.5 outperformed Claude 2 and Bard, showing strong capabilities in identifying relevant clinical concepts such as diagnosis codes, medications, and lab measurements, and translating them into CDM-compatible SQL structures. However, both GPT models struggled with constructing logically coherent inclusion/exclusion criteria, often producing algorithms that were either overly restrictive (low recall) or overly permissive (low PPV). Implementation of the top-rated LLM-generated algorithms demonstrated that expert review and refinement remain necessary to achieve clinically valid phenotypes comparable to the eMERGE gold-standard algorithms. The study highlights LLMs’ potential to accelerate early-stage phenotyping—particularly literature synthesis and initial rule drafting—while emphasizing the continued need for clinical and informatics expertise to ensure reliability and reproducibility in EHR-based research.

- **"Advancing Plant Biology Through Deep Learning–Powered Natural Language Processing"**, 2024  
  `plant-biology, protein-language-models, dna-llms, deep-learning, agricultural-bioinformatics`  
  — This perspective highlights the growing impact of deep learning—particularly large language models (LLMs) and protein language models (PLMs)—on plant biology and agricultural genomics. The authors describe how transformer-based PLMs enable large-scale representation learning directly from DNA and protein sequences, capturing multiscale structural, functional, and evolutionary patterns that traditional computational methods fail to model. These models support tasks such as regulatory motif identification, protein structure–function inference, variant impact prediction, and trait-associated sequence analysis. The article emphasizes their relevance for accelerating crop improvement, enabling model-guided genome editing, and uncovering biological mechanisms underlying complex plant traits. By integrating LLM-powered analyses with existing plant omics datasets, researchers can enhance predictions of gene function, understand plant cell systems, and design strategies for sustainable agroecological transitions. The authors argue that deep learning, anchored by LLMs and PLMs, will play a central role in future plant sciences, bridging molecular biology with large-scale agricultural applications while requiring continued human oversight due to ethical, regulatory, and interpretability considerations.

- **"WEFormer: Classification for physiological time series with small sample sizes based on wavelet decomposition and time series foundation models"**, 2025  
  `physiological-time-series, tsfm, wavelet-decomposition, small-sample-learning, transformer-models`  
  — This study introduces WEFormer, a physiological time-series classification framework designed specifically for scenarios with extremely limited sample sizes, a common challenge in biomedical research due to high data-collection cost, privacy restrictions, and difficulty recruiting subjects. The model integrates a pretrained Time Series Foundation Model (TSFM; MOMENT) with frozen weights as a universal feature extractor, allowing the architecture to leverage rich temporal representations without overfitting. In parallel, WEFormer incorporates a differentiable MODWT wavelet decomposition module that separates input signals into multi-frequency sub-bands; a learnable attention mechanism dynamically emphasizes informative frequency components while suppressing noise, enabling robust feature learning even from low-quality wearable-device signals such as GSR and ECG. The approach avoids task-specific priors required by meta-learning or prototypical methods and instead combines generalized pretrained embeddings with adaptive spectral filtering. Extensive experiments on two multimodal datasets—WESAD for emotion recognition and MOCAS for cognitive workload estimation—show substantial accuracy improvements over prior deep-learning and augmentation-based methods under small-sample conditions. Ablation studies confirm that TSFM embeddings and wavelet-attention decomposition are both central to WEFormer’s performance, demonstrating the value of foundation time-series models for practical clinical and affective-computing applications with limited data.

- **"Semi-supervised learning with pseudo-labeling compares favorably with large language models for regulatory sequence prediction"**, 2025  
  `regulatory-genomics, semi-supervised-learning, pseudo-labeling, cross-species-alignment, dnabert2-benchmarking`  
  — This work proposes a cross-species semi-supervised learning (SSL) framework that substantially expands training data for regulatory sequence prediction by pseudo-labeling homologous regions across mammalian genomes, addressing the fundamental limitation that supervised deep learning relies on scarce functional genomic labels constrained by the finite size of the human genome. The method remaps annotated regulatory sequences (e.g., TF-binding peaks) from human to closely related genomes, generating large-scale pseudo-labeled datasets for model pretraining, followed by fine-tuning on the original labeled human data. An enhancement inspired by the Noisy Student algorithm estimates pseudo-label confidence and particularly improves performance for transcription factors with very small training sets. The SSL paradigm is architecture-agnostic and was applied to DeepBind, DeepSEA, and DNABERT2, consistently yielding stronger sequence-classification accuracy and improved SNP-effect prediction across multiple TFs and assays. Notably, compact SSL-trained models achieved performance matching or surpassing large DNA language models such as DNABERT2, highlighting that data-efficient SSL pretraining can rival or outperform computationally expensive self-supervised LLMs trained on many genomes. The study demonstrates that evolutionary conservation provides a powerful signal for regulatory model scaling without the heavy resource cost of large LLM pretraining.

- **"From text to traits: exploring the role of large language models in plant breeding"**, 2024  
  `plant-breeding, plant-omics, llms, multimodal-integration, genetic-relationship-mining`  
  — This review examines how large language models, originally designed for natural language understanding, can be repurposed as computational engines for uncovering complex genetic, phenotypic, and environmental interactions in modern plant breeding. The paper outlines how LLMs and foundational transformer architectures can ingest heterogeneous biological information—multi-omics, field phenotyping, environmental metadata, and genomic variation—to reveal non-linear genotype–phenotype relationships, infer novel genetic interactions, and improve trait-performance prediction. By treating biological sequences, trait descriptions, and environmental conditions as structured "text," LLMs can build unified latent representations that enhance selection strategies compared to traditional quantitative genetics approaches. The author highlights the potential of LLM-driven knowledge graph construction, multimodal data fusion, and zero-shot reasoning to support breeders in decision-making, accelerate discovery of beneficial alleles, and enable more sustainable crop-improvement pipelines. Despite these opportunities, the review emphasizes challenges including limited plant-specific training corpora, data heterogeneity, scarcity of labeled phenotypic datasets, and risks of hallucinations, ultimately framing LLMs as an emerging but still underexplored direction for advancing computational plant breeding.

- **"Large AI Models in Health Informatics: Applications, Challenges, and the Future"**, 2023, JBHI  
  `foundation-models, health-informatics, multimodal-health-data, medical-ai, llms`  
  — This comprehensive review outlines the rise of large AI models—foundation models trained on massive multimodal biomedical datasets—and their transformative impact on health informatics. The authors analyze how these models, exemplified by ChatGPT-scale architectures, reshape methodologies across seven major sectors: bioinformatics, medical diagnosis, medical imaging, EHR-based informatics, medical education, public health, and medical robotics. The review highlights how transformer-based large models integrate heterogeneous biological and clinical data (genomics, imaging, structured EHRs, medical text), enabling improved prediction, reasoning, and decision-support across health workflows. It also discusses architectural advances, self-supervised pretraining approaches, and cross-modal representations that underpin the scalability and adaptability of foundation models in healthcare. Despite their rapid progress, the paper emphasizes critical challenges—including robustness, bias, hallucinations, data scarcity for specialized diseases, privacy constraints, and computational cost—while offering forward-looking insights on safe deployment, regulatory considerations, and the role of trustworthy, clinically aligned large models in the future of health informatics.

- **"Deciphering enzymatic potential in metagenomic reads through DNA language models"**, 2024  
  `dna-llms, metagenomics, enzymatic-annotation, foundation-models, read-level-analysis`  
  — This study introduces two transformer-based DNA language models—REMME, a foundational model pretrained on raw metagenomic reads to learn contextual DNA representations, and REBEAN, a fine-tuned enzymatic-function annotator that predicts enzyme activities directly from unassembled metagenomic reads. Unlike traditional reference-based pipelines that depend on sequence homology, curated databases, or assembled contigs, REBEAN identifies functional signatures in short reads by focusing on molecular function rather than strict gene identity. The models demonstrate strong generalization to both known and orphan sequences, uncovering functionally relevant subsequences even without explicit supervision. Importantly, REBEAN expands enzymatic annotation coverage for environmental metagenomes and enables the discovery of previously uncharacterized enzymes, offering a reference-free strategy for probing microbial functional “dark matter”.

- **"Natural language processing data services for healthcare providers"**, 2024  
  `clinical-nlp, ehr-text-mining, snomed-ct, ner-pipelines, healthcare-integration`  
  — This paper presents a first-of-its-kind clinical NLP service deployed within the UK National Health Service (NHS), designed as an integrated data-processing and annotation framework to align machine learning workflows with real clinical environments. Using harmonised parallel platforms, the authors developed a scalable infrastructure for clinical text annotation, data quality management, and model refinement. The system distils expert clinician knowledge into NLP models through continuous annotation cycles, resulting in more than **26,086 manual annotations across 556 SNOMED-CT concepts**. The service primarily leverages named entity recognition (NER) for extracting diagnoses, procedures, and clinical attributes from unstructured EHR text, enabling downstream operational and clinical decision-support applications. By embedding NLP capabilities directly into provider workflows, the approach improves data accessibility, informs analytics, and supports broader adoption of AI-driven healthcare solutions. The authors argue that such vertically integrated NLP services will soon become standard components of healthcare delivery infrastructures.

- **"Deciphering genomic codes using advanced NLP techniques: a scoping review"**, 2024  
  `genomics-nlp, dna-tokenization, transformers, llms, regulatory-annotation, dna-language-models`  
  — This scoping review synthesizes recent efforts applying Natural Language Processing (NLP) and transformer-based Large Language Models (LLMs) to genomic sequencing data analysis. Surveying 26 studies published between 2021 and 2024, the paper highlights how DNA tokenization strategies (k-mers, adaptive tokenizers, byte-pair encodings), together with transformer architectures, improve the representation of genomic sequences by capturing long-range dependencies and regulatory patterns. The reviewed models demonstrate strong performance in predicting functional genomic annotations, including transcription-factor binding, chromatin accessibility, enhancer/promoter states, and variant-effect inference. The authors emphasize that transformer-based genomics models enable scalable processing of large sequencing data and facilitate regulatory code interpretation, yet challenges remain in transparency, dataset accessibility, model bias, and biological interpretability. The review positions genomic NLP as a rapidly emerging field with significant potential to advance precision medicine through automated, high-resolution analysis of noncoding regulatory regions.

- **"Review and reconciliation: A proof-of-concept investigation"**, 2025  
  `clinical-llms, medication-review, drug-safety, pharmacogenomics, decision-support`  
  — This proof-of-concept study evaluates the ability of four large language models—ChatGPT, Gemini, Claude-Instant, and Llama—to support medication review and reconciliation workflows. The authors assessed LLM performance across key pharmacotherapy tasks, including detection of dosing-regimen errors, identification of drug–drug interactions, therapeutic-drug-monitoring (TDM)–based dose adjustments, and genomics-guided individualized dosing. Outputs were evaluated using predefined criteria (accuracy, relevance, risk-management behavior, hallucination control, and citation quality). Results show variable but generally consistent model behavior: ChatGPT demonstrated high accuracy in most dosing-error scenarios; all LLMs correctly identified warfarin-related interactions but collectively missed the clinically important metoprolol–verapamil interaction. Claude-Instant provided the most appropriate recommendations for TDM-based regimen adjustment and pharmacogenomic decision-making, while Gemini was notable for spontaneously including citations and guideline references, enhancing interpretability. Error-impact analysis revealed minor safety implications for dosing-regimen and TDM tasks, but potentially major consequences for missed drug–drug interactions or incorrect pharmacogenomic recommendations. The study highlights both the promise and current limitations of LLMs as medication-review assistants and underscores the need for validated integration into EHR and prescribing systems to ensure safe deployment in clinical workflows.

- **"Conversational AI agent for precision oncology: AI-HOPE-WNT integrates clinical and genomic data to investigate WNT pathway dysregulation in colorectal cancer"**, 2025  
  `precision-oncology, llm-agents, wnt-pathway, colorectal-cancer, clinical-genomics-integration`  
  — Introduces AI-HOPE-WNT, a conversational LLM-driven agent that performs natural-language analysis of WNT pathway dysregulation in colorectal cancer. The system integrates cBioPortal clinical and genomic datasets with automated statistical pipelines to enable mutation profiling, subgroup stratification, co-mutation discovery, and survival analysis. It successfully reproduces known WNT-pattern findings while uncovering new associations such as APC×FOLFOX survival effects and MSI–demographic interactions, demonstrating the utility of LLM-guided analytics for precision oncology.

- **"Foundation models for generalist medical artificial intelligence"**, 2023, Nature  
  `foundation-models, generalist-medical-ai, multimodal-llms, self-supervised-learning, medical-reasoning`  
  — Proposes the Generalist Medical AI (GMAI) paradigm where multimodal foundation models unify imaging, EHR, genomics, and clinical text through large-scale self-supervision. These models adapt to new tasks via natural-language instructions, supporting diagnostics, triage, and population health modeling. The paper outlines architectural requirements, multimodal pretraining, and safety challenges in deploying autonomous clinical reasoning systems.

- **"EDS-Kcr: Deep Supervision Based on Large Language Model for Identifying Protein Lysine Crotonylation Sites Across Multiple Species"**, 2024  
  `protein-llms, esm2, ptm-prediction, crotonylation, multi-species`  
  — Presents EDS-Kcr, a cross-species predictor for lysine crotonylation using ESM2 embeddings with deep supervision to capture subtle biochemical patterns. It integrates classical k-mer encodings with transformer features, outperforming existing ML and DL Kcr predictors. The model enhances interpretability and generalization across human, plant, and microbial proteins.

- **"Multiomics Research: Principles and Challenges in Integrated Analysis"**, 2024  
  `multi-omics, integration, deep-learning, gnn, gan, llms-in-biology`  
  — A comprehensive review of multiomics integration, covering genomics, transcriptomics, proteomics, metabolomics, and epigenomics. It discusses deep learning, GNNs, GANs, and LLMs for cross-modal feature extraction and biological insight generation. Key challenges include data heterogeneity, interpretability, and scalability of integrative models.

- **"Language Modelling Techniques for Analysing the Impact of Human Genetic Variation"**, 2024  
  `variant-effect-prediction, llms, transformers, dna-language-models, protein-lms`  
  — Reviews over fifty language-model approaches for predicting variant effects across DNA, RNA, and protein sequences. Highlights the shift from classical N-gram/RNN models to transformer-based LLMs that better capture regulatory and structural dependencies. Emphasizes the lack of unified benchmarks and the need for standardized evaluation frameworks.

- **"Multimodal Cell Maps as a Foundation for Structural and Functional Genomics"**, 2024  
  `multimodal-genomics, protein-mapping, llm-annotation, structural-genomics, proteomics`  
  — Builds a high-resolution multimodal map of >5,100 human proteins by integrating immunofluorescence imaging with AP–MS interaction data. LLMs annotate molecular assemblies, revealing structural complexes, new functional roles, and cancer-relevant assemblies. Validated using SEC–MS, establishing a foundational reference for cell organization.

- **"The PRIDE database at 20 years: 2025 update"**, 2025  
  `proteomics, mass-spectrometry, public-databases, FAIR-data, llm-tools`  
  — Summarizes major upgrades to the PRIDE proteomics repository, including large-scale MS reprocessing pipelines, improved metadata validation, and an LLM-powered chatbot for data discovery. Expanded support for DIA, crosslinking, and immunopeptidomics workflows, reinforcing PRIDE as a central FAIR-compliant proteomics resource.

- **"ADAM-1: An AI Reasoning and Bioinformatics Model for Alzheimer’s Disease Detection and Microbiome–Clinical Data Integration"**, 2025  
  `alzheimers-disease, llm, multi-agent-systems, microbiome, rag`  
  — Introduces ADAM-1, a multi-agent LLM framework integrating clinical metadata and gut microbiome profiles. Using RAG-supported reasoning, ADAM-1 outperforms XGBoost in stability and F1-score on 335 samples, offering interpretable biological explanations for AD-related microbial and immune patterns.

- **"DREAM: Autonomous Self-Evolving Research on Biomedical Data"**, 2025  
  `autonomous-research, llm, agentic-ai, automated-bioinformatics`  
  — Presents DREAM, a fully autonomous biomedical research agent capable of generating hypotheses, coding analyses, debugging, validating results, and iterating without human intervention. Outperforms expert scientists in question generation and achieves >10,000× efficiency improvement on Framingham Heart Study workflows.

- **"Transformers and Genome Language Models"**, 2024  
  `genome-language-models, transformers, dna-llms, sequence-modeling`  
  — Reviews transformer-based genomic language models (DNABERT, NT, HyenaDNA, MambaDNA) and their role in regulatory prediction, variant interpretation, and 3D genome modeling. Highlights attention scalability challenges and compares transformers with emerging state-space architectures.

- **"AI-Empowered Perturbation Proteomics for Complex Biological Systems"**, 2024  
  `perturbation-proteomics, foundation-models, systems-biology, deep-learning`  
  — Discusses how AI and foundation models can systematize perturbation proteomics through the PMMP (Perturbation → Measurement → Modeling → Prediction) pipeline. Highlights deep learning, GNNs, and generative models for capturing proteomic response patterns and inferring mechanisms of action in complex biological systems.

- **"Deep Learning Applications Advance Plant Genomics Research"**, 2025  
  `plant-genomics, deep-learning, gene-regulation, plant-LLMs, multi-omics`  
  — Reviews DL applications in plant genomics for regulatory modeling, protein prediction, and trait discovery. Covers CNNs, RNNs, transformers, GNNs, and plant-specific genome language models (e.g., PDLLMs, AgroNT). Emphasizes transfer learning and multi-omics integration for stress and trait analysis.

- **"Automatic Biomarker Discovery and Enrichment with BRAD"**, 2025  
  `agentic-LLMs, RAG, biomarker-discovery, enrichment-analysis`  
  — Introduces BRAD, an agentic LLM system that performs transparent, reproducible biomarker interpretation using RAG, ontology-aware retrieval, and automated enrichment reporting. Outperforms commercial chatbots in reliability, traceability, and biological grounding.

- **"Federated Deep Learning Enables Cancer Subtyping by Proteomics (ProCanFDL)"**, 2025  
  `federated-learning, proteomics, cancer-subtyping, privacy-preserving-AI`  
  — Demonstrates a federated deep-learning framework enabling multi-institutional cancer subtyping from DIA/TMT proteomics without sharing raw data. Achieves accuracy comparable to centralized models and +43% improvement over local-only training across 40 cohorts in 8 countries.

- **"Investigating the Prospects of ChatGPT in Training Medicinal Chemists and Novel Drug Development"**, 2025  
  `llms, medicinal-chemistry, drug-discovery, chemoinformatics`  
  — Reviews how ChatGPT accelerates medicinal chemistry workflows including literature synthesis, code generation, similarity analysis, ADMET reasoning, and education. Highlights benefits alongside limitations such as hallucinations, lack of multimodal chemical input, and reproducibility concerns.

- **"Integrating Multimodal Cancer Data using Deep Latent Variable Path Modelling (DLVPM)"**, 2025  
  `multimodal-integration, deep-learning, cancer-genomics, histopathology`  
  — Proposes DLVPM, a deep-learning path-modelling framework that integrates SNVs, methylation, miRNA, expression, and histopathology to infer cross-modal dependencies in cancer. Outperforms classical SEM and generalizes to scRNA-seq, CRISPR screens, and spatial transcriptomics.

- **"Deep Learning–Based Multimodal Biomedical Data Fusion: An Overview and Comparative Review"**, 2025  
  `multimodal-data, deep-learning, biomedical-fusion, transformers, gnns`  
  — A comprehensive survey of multimodal fusion strategies across imaging, omics, clinical text, and biosignals. Reviews CNNs, RNNs, transformers, GNNs, and generative models for cross-modal alignment. Highlights fusion levels (early, intermediate, late) and challenges in scalability and interpretability.

- **"AskBeacon — Performing Genomic Data Exchange and Analytics with Natural Language"**, 2025  
  `genome-analysis, beacon-protocol, llm-agents, secure-genomics`  
  — Introduces AskBeacon, an LLM interface for GA4GH Beacon servers enabling natural-language genomic queries without coding. Translates questions into secure Beacon-compliant queries, performs allele-frequency analysis, and generates publication-ready figures with strong privacy safeguards.

- **"Genotypic–Phenotypic Landscape Computation Based on First Principle and Deep Learning"**, 2025  
  `genotype-phenotype, fitness-landscape, transformers, viral-evolution`  
  — Develops the Phenotypic-Embedding theorem and a co-attention Transformer to compute genotype–fitness landscapes directly from SARS-CoV-2 sequences. Predicts immune-escape mutations, models epistasis and recombination, and derives viral fitness (R₀) from sequence alone.

- **"Multimodal Integration Strategies for Clinical Application in Oncology"**, 2025  
  `oncology, multimodal-integration, clinical-ai, multi-omics, imaging`  
  — Reviews multimodal AI approaches for integrating genomics, transcriptomics, proteomics, histopathology, radiology, EHRs, and spatial data in oncology. Highlights fusion architectures, vision–language models, and applications in prognosis, biomarker discovery, and treatment-response prediction.

- **"An Effective Encoding of Human Medical Conditions in Disease Space"**, 2025  
  `disease-embedding, multimodal-health-data, comorbidity-analysis, systems-biology`  
  — Proposes a unified embedding framework that maps diseases into a continuous latent “disease space,” capturing mechanistic and comorbidity relationships. Supports improved disease clustering, comorbidity discovery, and comorbidity-aware genetic analyses using multimodal clinical and genetic data.

- **"Kolmogorov–Arnold Networks for Genomic Tasks"**, 2025  
  `KANs, genomic-deep-learning, regulatory-genomics, sequence-classification`  
  — Evaluates Linear and Convolutional KANs for genomic sequence modeling across promoter, enhancer, TF-binding, and flipon datasets. LKANs outperform CNNs/MLPs with fewer parameters and show strong potential as lightweight alternatives to transformer-based genomic models.

- **"Artificial Intelligence in Healthcare: A Survival Guide for Internists"**, 2025  
  `medical-ai, clinical-decision-support, llms, diagnostics`  
  — Provides a practical overview for clinicians on discriminative vs generative AI, LLM capabilities, hallucination risks, evaluation metrics, and safe deployment in internal medicine. Emphasizes AI’s potential in diagnostics and precision medicine while warning about biases and data quality issues.

- **"Pre-Meta: Priors-Augmented Retrieval for LLM-Based Metadata Generation"**, 2025  
  `genomics-metadata, RAG, ontology-integration, data-harmonization`  
  — Introduces Pre-Meta, a priors-augmented RAG pipeline that improves genomic metadata annotation by integrating ontology terms and structured priors. Enhances accuracy across models without fine-tuning and reduces LLM hallucinations in dataset metadata generation.

- **"BV-BRC: A Unified Bacterial and Viral Bioinformatics Resource with Expanded Functionality and AI Integration"**, 2025  
  `pathogen-genomics, bacterial-viruses, bioinformatics-platforms, RAG, AI-copilots`  
  — Presents BV-BRC, a comprehensive pathogen genomics ecosystem hosting >14M genomes with multi-omics integration. The update introduces an LLM-powered Copilot for natural-language analysis, improved pipelines for assembly and metagenomics, and FAIR-compliant workflows for bacterial and viral research.

- **"PaxDb v6.0: Reprocessed, LLM-Selected, Curated Protein Abundance Data Across Organisms"**, 2025  
  `proteomics, protein-abundance, mass-spectrometry, data-curation, LLM-classification`  
  — Major update to PaxDb with 1639 datasets across 392 species, integrating large-scale MS reprocessing via FragPipe and LLM-backed dataset selection. Provides harmonized healthy-state protein abundance maps enabling robust cross-species comparisons and FAIR proteomics data reuse.

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
