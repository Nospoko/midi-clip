# midi-clip

To run training:

```
python train.py --config-name <config>
```

# Model blueprints

### CLIP Training

```mermaid
flowchart TD
    A[MIDI Sequence] --> B(Pitch)
    A --> C(Velocity+Time)
    B --> |encoder| D(pitch embeddings)
    C --> |encoder| E(vt embeddings)
    D:::trainable --> F(CLIP LOSS)
    E:::trainable --> F
    classDef trainable stroke:teal
```

### Pitch Encoder

```mermaid
flowchart TD
    A[MIDI Sequence] --> B(Pitch)
    B --> C[nn.Embedding]
    D[SinusoidalPositionEmbeddings] --> E[+]
    C --> E
    E --> |attention blocks| F[embedding]
    F --> |mean| G[embedding]
    G --> |output projection| H[output]
```

### Velocity-Time Encoder

```mermaid
flowchart TD
    A[MIDI Sequence] --> |quantization| B(Quantized Piece)
    B --> C[velocity embedding]
    B --> D[dstart embedding]
    B --> E[duration embedding]
    C --> F[cat embedding]
    D --> F 
    E --> F
    F --> G(linear projection) --> I[embedding]
    H[SinusoidalPositionEmbeddings] --> I
    I --> J(attention blocks)
    J --> K(mean)
    K --> OUT(output projection)
```

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
