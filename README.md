# LLM-powered Driving Intelligence Evaluation Framework

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper:  
**"A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation"**  
*[Authors] | [Conference/Journal] | [Year]*

## 🚀 Framework Overview

This repository contains the core components for automating driving intelligence evaluation using LLMs:
'''
.
├── driving_context_to_des.py # Preprocessing: Driving context summarization
├── evaluation_rag_auto.py # Main evaluation pipeline
└── (Additional components pending upload)
'''


## 🔧 Key Components

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
