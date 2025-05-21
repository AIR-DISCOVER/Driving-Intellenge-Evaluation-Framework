# LLM-powered Driving Intelligence Evaluation Framework

Official implementation of the paper:  
**"A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation"**  

# **Work in Progress**: Full implementation will be uploaded shortly

## Framework Overview

This repository contains the core components for automating driving intelligence evaluation using LLMs:
'''
.
├── driving_context_to_des.py # Preprocessing: Driving context summarisation
├── evaluation_rag_auto.py # Main evaluation pipeline
└── (Additional components pending upload)
'''


##  Key Components

### 1. Driving Context Processor (`driving_context_to_des.py`)
- **Function**: Ingests raw driving context data and generates structured summaries
- **Input**: Time-series driving signals (steering, acceleration, etc.)
- **Output**: Natural language descriptions for LLM processing
- **Features**:
  - Event detection
  - Temporal pattern extraction

### 2. Automated Evaluation Engine (`evaluation_rag_auto.py`)
- **Function**: Performs end-to-end driving intelligence assessment
- **Core Methods**:
  - RAG-based knowledge retrieval from driving manuals
  - Multi-aspect evaluation (safety, intelligence, comfort)
  - Explainable scoring with LLM-generated feedback
- **Output Formats**:
  - Quantitative scores (0-10 scale)
  - Qualitative improvement suggestions
  - Evaluation reports
