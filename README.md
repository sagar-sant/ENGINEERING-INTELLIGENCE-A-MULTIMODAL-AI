# Engineering Intelligence - A Multimodal AI 

### A unified AI Framework for Engineering, Technical, Scientific, Mathematical Analysis and Intelligent Assistance

> A unified AI framework for engineering, technical, scientific, and mathematical analysis and intelligent assistance.

---

## Overview

**Engineering Intelligence** is a modular, multimodal artificial intelligence framework designed to provide a unified environment for engineering, technical, scientific, and mathematical problem solving.

The framework combines **natural language processing, locally deployed large language models, intelligent model routing, retrieval-augmented generation (RAG), document intelligence, deterministic numerical and symbolic computation, multimodal vision, LangChain-based AI application components, and LangGraph-based workflow orchestration** within a single interactive application.

Rather than relying on a single language model for every task, the system analyzes the nature of a request and routes it to an appropriate language model, retrieval pipeline, deterministic computational tool, or multimodal processing capability.

The framework is designed to support:

- Engineering and scientific question answering
- Mathematical and symbolic computation
- Technical document analysis
- Retrieval-augmented question answering
- Engineering drawings and schematics
- Technical images and figures
- Natural-language technical conversations
- Context-aware document reasoning
- Multimodal technical analysis

---

# Key Features

## 1. Natural Language Processing and Large Language Models

Natural language processing forms the primary interaction layer of Engineering Intelligence.

The framework supports multiple locally deployed large language models and provides capabilities including:

- Conversational question answering
- Technical and scientific reasoning
- Natural-language query understanding
- Request classification
- Task-dependent model selection
- Structured technical response generation

Local model inference is provided through **Ollama**, enabling the framework to work with multiple locally available model families.

Examples of model families used during development and testing include:

- Gemma
- Qwen
- DeepSeek
- Llama
- Mistral
- Phi
- Vision-capable Qwen models

The application is designed to remain model-agnostic at the routing layer, allowing different models to be selected according to task requirements.

---

## 2. Intelligent Model Routing

Engineering problems have different reasoning and computational requirements.

The framework therefore includes an intelligent model-routing layer that analyzes the characteristics of a request and determines the most appropriate processing path.

```text
User Request
     |
     v
Request Analysis
     |
     +----> Conversation / Technical QA
     |
     +----> Advanced Reasoning
     |
     +----> Document Retrieval / RAG
     |
     +----> Deterministic Calculation
     |
     +----> Multimodal Vision

This design avoids forcing every request through a single language model and provides a foundation for task-specific model selection.

## 3. Document Intelligence and Retrieval-Augmented Generation

Technical documents can be processed and queried through a retrieval-augmented generation pipeline.

The document workflow includes:

Document
   |
   v
Ingestion
   |
   v
Text Extraction
   |
   v
Chunking
   |
   v
Embedding Generation
   |
   v
Vector Storage
   |
   v
Relevant Retrieval
   |
   v
Context-Aware LLM Response


This allows responses to be grounded in information retrieved from uploaded technical documents rather than relying exclusively on pretrained language-model knowledge.

The document pipeline is designed for use cases such as:

Engineering manuals
Technical specifications
Procedures
Tables
Technical reports
Scientific documents
Engineering PDFs
Structured and semi-structured technical content

4. Deterministic Mathematical and Engineering Computation

A dedicated computational layer is used for numerical and symbolic problems.

The framework integrates:

Python
NumPy
SciPy
SymPy

A calculation detector identifies quantitative requests and routes suitable problems to a deterministic computational pathway. This design separates language reasoning from numerical execution and reduces dependence on language-model arithmetic for supported mathematical and engineering calculations.

5. Calculation Planning

More complex engineering and mathematical problems can be transformed into structured computational plans before deterministic execution.

Natural-Language Problem
          |
          v
Problem Analysis
          |
          v
Calculation Plan
          |
          v
Deterministic Execution
          |
          v
Verified Result
          |
          v
Technical Explanation

The calculation-planning layer provides a structured bridge between natural-language problem descriptions and deterministic computational tools.

6. Multimodal Engineering Analysis

Engineering Intelligence extends beyond text-only interaction through multimodal vision capabilities.

The framework is designed to process technical visual information such as:

Engineering drawings
Schematics
Diagrams
Technical figures
Component images
Visual engineering information

A multimodal request can combine visual information with natural-language instructions so that the system can interpret the image and produce a technical response.

Multimodal Workflow

Technical Image
      +
Natural-Language Instruction
      |
      v
Multimodal Model
      |
      v
Technical Interpretation
      |
      v
Structured Response

7. LangChain and LangGraph

The project uses LangChain and LangGraph as complementary components within the AI application architecture.

LangChain

LangChain is used as part of the LLM application layer for integrating language-model-driven components and AI application workflows.

LangGraph

LangGraph provides the structured orchestration layer, representing processing stages as nodes with conditional routing between them.

Combined Architecture

User Input
    |
    v
Input Analysis
    |
    v
LangGraph Routing
    |
    +--------------+--------------+--------------+
    |              |              |              |
    v              v              v              v
   LLM            RAG       Deterministic    Multimodal
 Reasoning                  Computation       Vision
    |              |              |              |
    +--------------+--------------+--------------+
                           |
                           v
                  Response Assembly
                           |
                           v
                      Streamlit UI

The orchestration architecture allows the system to support different execution paths while maintaining a unified application interface.

8. Local-First AI Architecture

A major design characteristic of the framework is the use of locally deployed model inference.

Language and vision models are accessed through Ollama, allowing multiple models to coexist within the development environment.

This provides:

Local model execution
Flexible model selection
Reduced dependence on external inference APIs for core processing
Support for multiple model families
A modular foundation for future model integration
9. Engineering Application Interface

The complete framework is exposed through an interactive Streamlit application.

The interface provides access to:

Engineering documents
Drawings and schematics
Mathematics and engineering calculations
Technical question answering
Natural-language interaction
Multimodal technical analysis

The application also supports structured response presentation and mathematical notation.

Technology Stack
Development Environment
Visual Studio Code
Python
AI and NLP
Large Language Models
Natural Language Processing
Ollama
LangChain
LangGraph
Retrieval and Document Intelligence
Retrieval-Augmented Generation (RAG)
Embeddings
Vector retrieval
Document extraction
Document chunking
Document ingestion
Engineering and Scientific Computation
NumPy
SciPy
SymPy
Multimodal Processing
Vision-capable language models
Technical image interpretation
Application
Streamlit


Project Architecture

The source code is organized as a modular Python application.

src/
├── app/
│   └── Application and document-service logic
│
├── chunking/
│   └── Document chunking
│
├── embeddings/
│   └── Embedding generation
│
├── evaluation/
│   └── Evaluation and testing components
│
├── extraction/
│   └── Document and content extraction
│
├── ingestion/
│   └── Document ingestion workflows
│
├── llm/
│   ├── model_router.py
│   ├── openai_llm.py
│   └── ollama/
│       ├── ollama_llm.py
│       └── __init__.py
│
├── multimodal/
│   └── Vision and multimodal processing
│
├── orchestration/
│   └── engineering_graph.py
│
├── pipeline/
│   └── Processing pipelines
│
├── rag/
│   └── Retrieval-augmented generation
│
├── retrieval/
│   └── Retrieval components
│
├── tools/
│   ├── calculator.py
│   └── calculation_planner.py
│
├── utils/
│   └── Utility components
│
└── vectorstore/
    └── Vector storage and retrieval infrastructure


End-to-End Workflow

A typical request follows a task-dependent processing workflow:

User Query / Document / Image
            |
            v
     Input Processing
            |
            v
     Request Analysis
            |
            v
     Intelligent Routing
            |
     +------+--------+-------------+
     |      |        |             |
     v      v        v             v
    LLM    RAG   Calculator      Vision
     |      |        |             |
     +------+--------+-------------+
                  |
                  v
         Response Generation
                  |
                  v
             Streamlit UI


Example Use Cases

Engineering Question Answering
What is the relationship between voltage,
current and resistance?
Mathematical Computation
What is the cube of 7425?

Expected deterministic result:

409,344,890,625
Technical Document Question Answering
Upload a technical PDF and ask:

"What is the operating temperature specified
for the component?"
Engineering Visual Analysis
Upload a schematic and ask:

"Explain the components and trace the
conventional current path."
Scientific Reasoning
Explain the relationship between pressure,
volume and temperature for an ideal gas.

Functional Evaluation

The framework has been functionally tested across its major processing capabilities.

| Capability               | Evaluation                         | Result        |
| ------------------------ | ---------------------------------- | ------------- |
| Conversational AI        | General conversational queries     | Successful    |
| Technical QA             | Engineering and scientific queries | Successful    |
| Mathematical computation | (7425^3)                           | Exact result  |
| Calculation routing      | Quantitative query detection       | Correct route |
| Calculation planning     | Structured computational problem   | Successful    |
| Document intelligence    | Document-service integration       | Successful    |
| Model routing            | Multiple query categories          | Successful    |
| LangGraph orchestration  | Workflow execution                 | Successful    |
| Multimodal processing    | Technical visual inputs            | Successful    |
| Streamlit interface      | End-to-end application testing     | Successful    |


The evaluation focuses primarily on functional integration, routing behavior, deterministic computation, and end-to-end application execution.

Design Philosophy

The central design principle of Engineering Intelligence is:

Use language models for language and reasoning, and use specialized tools for operations that benefit from deterministic computation, retrieval, or multimodal processing.

The resulting architecture combines complementary capabilities rather than treating a single language model as a universal solution.

Project Status

Status: Functional prototype / research project

The core framework has been implemented and functionally tested across conversational, computational, document-based, retrieval, multimodal, orchestration, and application workflows.


Author

Sagar Sant

BSc. (Hons.) Data Science & Artificial Intelligence
Indian Institute of Technology Guwahati



