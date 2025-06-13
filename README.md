# kani-multimodal-core

This package contains core extensions for using Kani with multimodal language models.

## Features

This package provides the core multimodal extensions that engine implementations can use -- it does not provide any
engine implementations on its own.

The package adds support for:

- Images
- Audio
- Video (WIP)

When installed, these core kani engines will automatically use the multimodal parts:

- OpenAIEngine
- AnthropicEngine

## Installation

kani-multimodal-core should be installed alongside the core kani install using an extra:

```shell
$ pip install "kani[multimodal]"
```

However, you can also explicitly specify a version and install the core package itself:

```shell
$ pip install kani-multimodal-core
```

TODO more docs here and readthedocs
