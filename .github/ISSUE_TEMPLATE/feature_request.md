---
name: Feature request
about: Propose a capability or an improvement
labels: enhancement
body:
  - type: textarea
    id: problem
    attributes:
      label: The problem
      description: What are you trying to do that Blazar cannot do today, or does awkwardly? Describe the job, not the solution.
    validations:
      required: true
  - type: textarea
    id: current-workaround
    attributes:
      label: How do you work around it today?
  - type: textarea
    id: proposal
    attributes:
      label: Proposed shape (optional)
      description: API / CLI shape you'd expect. Open to alternative designs solving the same problem.
---
