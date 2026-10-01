---
name: Bug report
about: Something broke or behaved wrong
labels: bug
body:
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear description of the bug. Include the exact command or request (curl body) and the full error output.
      placeholder: |
        POST /v1/chat/completions with ... returned 500 ...
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: What did you expect?
    validations:
      required: true
  - type: textarea
    id: reproduce
    attributes:
      label: Steps to reproduce
      placeholder: |
        1. blazar serve with config ...
        2. curl ...
        3. ...
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: Output of `blazar doctor` (or at minimum: OS, GPU + VRAM, blazar version from `blazar --version`, model + quant).
      placeholder: |
        OS: ...
        GPU: ...
        blazar: 0.15.0
        model: ...
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant logs
      description: '`journalctl -u blazar -n 200 --no-pager` for the failing window, and `blazar why <trace>` when a trace id appears in the error.'
---
