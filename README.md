# REA-Coder

REA-Coder is an innovative, multi-agent collaborative framework inspired by software requirement engineering, designed to tackle complex code generation tasks. By simulating a rigorous software development lifecycle, REA-Coder effectively translates highly ambiguous or complex problem descriptions into robust, executable code.

## The REA-Coder Framework

Unlike standard direct-prompting methods, REA-Coder decomposes the code generation problem into a pipeline of specialized agents. This multi-agent architecture ensures that requirements are fully understood, edge cases are identified, and logic gaps are bridged before the final code is generated.

The framework consists of four primary agents:

- **Agent 1: Clarification (Question Generation)**
  - **Role:** Analyzes the initial problem description to identify ambiguities, missing constraints, or unclear logic.
  - **Action:** Generates targeted clarifying questions and produces "gold answers" to establish a solid foundation for the problem.
- **Agent 2: Requirement Engineering**
  - **Role:** Acts as the system architect.
  - **Action:** Summarizes logic gaps, formalizes the requirements, and generates "stop test cases" to prevent infinite loops and ensure edge-case handling.
- **Agent 3: Code Generation**
  - **Role:** The primary developer.
  - **Action:** Ingests the clarified requirements and answers from the previous agents to synthesize the final, robust code solution. It also includes mechanisms for filling masked requirements to ensure all constraints are met.
- **Agent 4: Evaluation and Feedback**
  - **Role:** The reviewer and tester.
  - **Action:** Evaluates the generated code against the initial and refined requirements (mask recovery evaluation) and provides structured feedback for iterative refinement.

## Repository Structure

The project is structured to separate the core REA-Coder framework from the comparative baseline methods.

```
.
├── REA-Coder/                      # Core implementation of the REA-Coder framework
│   ├── apps_controller/            # Multi-agent logic and orchestration
│   │   ├── agent1_questions.py     # Clarification and question generation
│   │   ├── agent2_requirements.py  # Requirement engineering and gap summarization
│   │   ├── agent3_codegen.py       # Code generation and answer synthesis
│   │   ├── agent4_feedback.py      # Evaluation and feedback generation
│   │   ├── prompts_markdown/       # System prompts for all agents
│   │   └── run_eval.py             # Main execution script
│   ├── apps_eval/                  # Execution and evaluation environment
│   │   ├── executor.py             # Code execution sandbox
│   │   └── parallel_runner.py      # Parallel evaluation runner
│   └── llm/                        # Large Language Model client definitions
│       └── client.py               # API request handlers and routing
└── baseline/                       # Implementations of comparative baselines
    ├── Icot/                       # Interactive Chain-of-Thought
    ├── Scot/                       # Structured Chain-of-Thought
    ├── Self-collaboration/         # Multi-role collaborative framework
    ├── Self-repair/                # Execution-based automated repair
    ├── Specfix/                    # Specification-driven generation
    ├── Specine/                    # Alignment and sanitization focus
    └── ufix/                       # Fix generation using CodeBLEU and feedback
```

## Supported Models

The framework integrates a flexible LLM client interface configured to support the following advanced models for generation and routing:

- **qwen3-coder-30b-a3b-instruct**
- **gpt-5-mini-2025-08-07**
- **gemini-3-flash-preview**
- **deepseek-chat**

## Evaluated Datasets

REA-Coder is designed to be evaluated on a comprehensive suite of programming benchmarks to test functional correctness under varying levels of complexity:

- **apps**
- **Codecontests_raw**
- **Codecontests**
- **XCodeEval**
- **livecodebench**

## Comparative Baselines

To demonstrate the efficacy of REA-Coder, this repository includes integrated implementations of several recent baseline methods for code generation and self-repair. These are provided for brief comparison:

- **ICoT:** Interactive Chain-of-Thought.
- **SCoT:** Structured Chain-of-Thought.
- **Self-collaboration:** A multi-role (analyst, coder, tester) LLM collaboration framework.
- **Self-repair:** Automated code repair using execution feedback.
- **Specfix & Specine:** Specification-driven generation and sanitization approaches.
- **UFix:** Evaluation and fix-generation using CodeBLEU and execution feedback.

## Getting Started

### Installation

Ensure you have Python 3.9+ installed. Clone the repository and install the dependencies.

```
git clone https://anonymous.4open.science/r/REA-Coder-6CFE
cd REA-Coder
pip install -r requirements.txt
```

### Running REA-Coder

You can execute the main REA-Coder evaluation pipeline by running the controller script:

```
python REA-Coder/apps_controller/run_eval.py
```

To configure API keys or switch between models, please update the settings within `REA-Coder/llm/client.py` and the respective routing configurations.