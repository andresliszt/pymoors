# Installation

<div class="lang-rust" markdown>

{% include-markdown "getting_started/rust/installation.md" %}

</div>

<div class="lang-python" markdown>

{% include-markdown "getting_started/python/installation.md" %}

</div>


# Development

## Quick Start

```sh
# Fork & clone the repo
git clone https://github.com/your-username/moo-rs.git
cd moo-rs

# Setup development environment (syncs dependencies & hooks)
make setup

# Build everything (moors + pymoors)
make build-dev

# Run all tests
make test

# Format & lint code
make lint
make fmt
```

For detailed requirements and advanced commands, see the [Developer Guide](../development/setup_environment.md).
