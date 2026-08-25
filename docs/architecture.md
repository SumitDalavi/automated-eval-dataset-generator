# Architecture: Automated Eval Dataset Generator

## System Diagram
The following Mermaid.js sequence diagram maps the core workflow and interactions:

```mermaid
sequenceDiagram
ProdLogs->>Ingestion: Raw traces
Ingestion->>HDBSCAN: Cluster by semantics
HDBSCAN->>LLMJudge: Grade quality
LLMJudge->>Dataset: Insert as positive/negative eval case
```

## Component Breakdown
- **Core Technology**: Python, scikit-learn, HDBSCAN, ClickHouse
- **Design Paradigm**: Emphasizes high availability, fault tolerance, and security.
