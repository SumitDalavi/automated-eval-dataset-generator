# automated-eval-dataset-generator Architecture
> Maturity: Functional Prototype

## System Diagram
The following Mermaid.js sequence diagram maps the core workflow and interactions within the system:

```mermaid
sequenceDiagram
    SourceDoc->>Parser: Extract Text
Parser->>LLM: Generate QA Pairs
LLM-->>Generator: QA JSON
Generator->>Formatter: Convert to JSONL
Formatter-->>Storage: Save Dataset
```

## Component Breakdown
- **Core Technology**: Python, HuggingFace Datasets
- **Design Paradigm**: Emphasizes high availability, fault tolerance, and security boundaries.

## Security & Scaling Considerations
- Strict input validations and sanitization.
- Horizontal scalability achieved via stateless workers and queues where applicable.
- Encrypted data at rest and in transit.
